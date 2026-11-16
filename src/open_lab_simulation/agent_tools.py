"""Run-local pinned registrations; source is data here and executes only in python.run."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from jsonschema import Draft202012Validator

from .paths import is_link
from .registry import tokenize

PERMISSIONS = {"filesystem": "workspace_write", "network": "none", "subprocess": True}
CREATE = {"interface": "mcp", "server": "runtime-tools", "tool": "tools.create"}
MAX_TOOLS = 16
MAX_SOURCE = 32_000
MAX_DATA = 16_000
WRAPPER = '''import json, sys
request = json.load(sys.stdin)
scope = {"__name__": "agent_tool", "__file__": "<pinned-agent-tool>"}
exec(compile(request["source"], "<pinned-agent-tool>", "exec"), scope)
result = scope["run"](request["arguments"])
print(json.dumps(result, allow_nan=False))
'''


def bounded_json(value, limit=MAX_DATA):
    try:
        text = json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (ValueError, TypeError, RecursionError):
        raise ValueError("tool data must be finite JSON") from None
    if len(text.encode("utf-8")) > limit:
        raise ValueError(f"tool JSON exceeds {limit} bytes")
    return text


def safe_schema(schema):
    """A small finite subset: no references, regex, compositions or remote retrieval."""
    bounded_json(schema)
    nodes = 0

    def visit(node, depth=0):
        nonlocal nodes
        nodes += 1
        if not isinstance(node, dict) or depth > 8 or nodes > 128:
            raise ValueError("tool schema exceeds depth/node limits")
        kind = node.get("type")
        allowed = {"type", "description", "enum"}
        if kind == "object":
            allowed |= {"properties", "required", "additionalProperties"}
            if node.get("additionalProperties") is not False:
                raise ValueError("tool object schemas require additionalProperties=false")
            properties = node.get("properties", {})
            if not isinstance(properties, dict) or len(properties) > 64:
                raise ValueError("tool object has too many properties")
            for child in properties.values():
                visit(child, depth + 1)
        elif kind == "array":
            allowed |= {"items", "minItems", "maxItems"}
            if type(node.get("maxItems")) is not int or not 0 <= node["maxItems"] <= 1024:
                raise ValueError("tool arrays require maxItems <= 1024")
            visit(node.get("items"), depth + 1)
        elif kind == "string":
            allowed |= {"minLength", "maxLength"}
            if type(node.get("maxLength")) is not int or not 0 <= node["maxLength"] <= MAX_DATA:
                raise ValueError("tool strings require maxLength <= 16000")
        elif kind in ("integer", "number"):
            allowed |= {"minimum", "maximum"}
        elif kind not in ("boolean", "null"):
            raise ValueError("unsupported tool schema type")
        if set(node) - allowed:
            raise ValueError("unsupported tool schema keyword")
        if "enum" in node and (not isinstance(node["enum"], list) or len(node["enum"]) > 64):
            raise ValueError("tool enum exceeds 64 choices")

    visit(schema)
    if schema.get("type") != "object":
        raise ValueError("tool inputs and outputs must be objects")
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as error:
        raise ValueError("invalid tool JSON schema") from error


def check(schema, data):
    bounded_json(data)
    if not Draft202012Validator(schema).is_valid(data):
        raise ValueError("tool data does not satisfy its schema")


class AgentTools:
    def __init__(self, host, run_id, contracts, records=None):
        self.host, self.run_id, self.contracts = host, run_id, contracts
        self.records = {}
        if records is not None:
            if not isinstance(records, dict) or len(records) > MAX_TOOLS:
                raise ValueError("invalid checkpoint tool registrations")
            for ident, record in records.items():
                if set(record) != {"definition", "source", "sha256"}:
                    raise ValueError("invalid checkpoint tool fields")
                definition, source = record["definition"], record["source"]
                self.validate(definition, source)
                if ident != definition["id"] or record["sha256"] != self.digest(source):
                    raise ValueError("checkpoint tool digest or id differs")
                self.install(definition, source)

    @staticmethod
    def digest(source):
        return hashlib.sha256(source.encode("utf-8")).hexdigest()

    def validate(self, definition, source):
        manifest = self.host.capabilities["tools.create"][1]
        if not Draft202012Validator(manifest["input_schema"]).is_valid(definition):
            raise ValueError("invalid tool definition")
        ident = definition["id"]
        if not re.fullmatch(r"agent\.[a-z][a-z0-9-]{0,55}", ident):
            raise ValueError("tool ids must use agent.<lower-case-slug>")
        if ident in self.host.capabilities:
            raise ValueError("tool id already exists; choose a new id")
        if not isinstance(source, str) or not 1 <= len(source.encode("utf-8")) <= MAX_SOURCE:
            raise ValueError("tool source exceeds 32000 bytes or is empty")
        safe_schema(definition["input_schema"])
        safe_schema(definition["output_schema"])
        for example in definition["examples"]:
            check(definition["input_schema"], example["input"])
            check(definition["output_schema"], example["expected_output"])

    def source(self, value):
        parts = value.split("/")
        reserved = {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"}
        reserved.update(f"{prefix}{n}" for prefix in ("COM", "LPT") for n in range(1, 10))
        if (len(parts) > 8 or any(not re.fullmatch(r"[A-Za-z0-9_-]+(?:\.py)?", p)
                                  or p.split(".")[0].upper() in reserved for p in parts)
                or not value.endswith(".py")):
            raise ValueError("source_path must be a relative workspace .py file")
        root = self.host.workspace.resolve(strict=True)
        target = root
        for part in parts:
            target = target / part
            if is_link(target):
                raise ValueError("tool source links are refused")
        if not target.resolve().is_relative_to(root) or not target.is_file():
            raise ValueError("tool source must be a workspace file")
        if target.stat().st_nlink != 1:
            raise ValueError("tool source hard links are refused")
        try:
            with target.open("rb") as handle:
                data = handle.read(MAX_SOURCE + 1)
            if len(data) > MAX_SOURCE:
                raise ValueError("tool source exceeds 32000 bytes")
            return data.decode("utf-8")
        except (OSError, UnicodeError):
            raise ValueError("cannot read UTF-8 tool source") from None

    def execute(self, definition, source, arguments):
        # All user source, including module-level code, runs in a fresh sandbox.
        check(definition["input_schema"], arguments)
        outcome = self.host._run("python.run", {
            "code": WRAPPER, "stdin": bounded_json({"source": source, "arguments": arguments},
                                                    100_000), "timeout_s": 5})
        if outcome.get("status") != "ok":
            raise ValueError(f"sandbox execution failed: {outcome.get('error', 'unknown')}")
        result = outcome["result"]
        if result["timed_out"]:
            raise ValueError("agent tool timed out")
        if result["exit_code"] != 0:
            raise ValueError(f"agent tool execution failed: {result['stderr'][:2000]}")
        if result["stdout_truncated"] or result["stderr_truncated"]:
            raise ValueError("agent tool output was truncated")
        try:
            payload = json.loads(result["stdout"])
        except (ValueError, RecursionError):
            raise ValueError("agent tool must return JSON without extra stdout") from None
        check(definition["output_schema"], payload)
        return payload

    def install(self, definition, source):
        ident = definition["id"]
        digest = self.digest(source)
        demos = [{"id": f"example-{i}", "title": f"Validated example {i}", **example}
                 for i, example in enumerate(definition["examples"], 1)]
        manifest = {
            "schema": "open-lab/capability", "schema_version": "1.0.1", "id": ident,
            "version": "0.1.0", "name": definition["name"], "summary": definition["summary"],
            "description": definition["description"], "categories": ["agent-tools"],
            "keywords": definition.get("keywords", []),
            "invocation": {"interface": "mcp", "server": "runtime-tools", "tool": ident},
            "input_schema": definition["input_schema"],
            "output_schema": definition["output_schema"],
            "permissions": dict(PERMISSIONS), "resources": {"timeout_s": 5},
            "demonstrations": demos, "self_test": {"demonstrations": [d["id"] for d in demos]},
            "limitations": ["Pinned source snapshot; workspace source edits do not replace it.",
                            "Five-second sandbox execution; examples are not general correctness."],
            "failure_modes": [], "provenance": {"origin": "agent", "created_in_run": self.run_id},
            "extensions": {"source_sha256": digest, "source_path": definition["source_path"]},
        }
        problems = self.contracts.validate(manifest, "capability")
        if problems:
            raise ValueError("generated agent manifest is invalid")
        self.records[ident] = json.loads(json.dumps({"definition": definition, "source": source,
                                                   "sha256": digest}))
        self.host.capabilities[ident] = (Path(), manifest)
        if self.host.registry is not None:
            entry = {k: manifest[k] for k in ("id", "name", "summary", "categories", "keywords")}
            entry["tokens"] = {k: sorted(tokenize(" ".join(v) if isinstance(v, list) else v))
                               for k in ("id", "name", "summary", "categories", "keywords",
                                         "description") for v in [manifest[k]]}
            entry["documentation"] = json.dumps(manifest, ensure_ascii=False, indent=2)
            entry["provenance"] = manifest["provenance"]
            self.host.registry.entries[ident] = entry
        return manifest

    def create(self, arguments):
        if len(self.records) >= MAX_TOOLS:
            raise ValueError("at most sixteen agent tools per run")
        source = self.source(arguments["source_path"])
        self.validate(arguments, source)
        for example in arguments["examples"]:
            actual = self.execute(arguments, source, example["input"])
            if actual != example["expected_output"]:
                raise ValueError("agent tool example differs from expected_output")
        manifest = self.install(arguments, source)
        return {"id": manifest["id"], "registered": True,
                "source_sha256": self.digest(source), "examples_passed": len(arguments["examples"]),
                "provenance": manifest["provenance"]}

    def invoke(self, ident, arguments):
        record = self.records[ident]
        payload = self.execute(record["definition"], record["source"], arguments)
        return {"status": "ok", "result": payload, "extensions": {
            "provenance": self.host.capabilities[ident][1]["provenance"],
            "source_sha256": record["sha256"]}}
