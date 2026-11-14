"""The tool host: the only way an agent's tool call reaches a capability.

Implements the invocation protocol specified in the Builder's
``docs/architecture/CONTRACTS.md`` ("Capability"). Every call is recorded
before anything runs and every outcome is recorded after, whatever happens.

Authority comes from the experiment configuration alone. Nothing in the
arguments, and nothing the model says, can widen what a call may do.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from .paths import minimal_environment
from .registry import META_SCHEMAS, Registry

DEFAULT_TIMEOUT_S = 30
_RUNNER = Path(__file__).with_name("_capability_runner.py")
_SLUG = re.compile(r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$")
_FILESYSTEM = {"none": 0, "workspace_read": 1, "workspace_write": 2}
_NETWORK = {"none": 0, "loopback": 1, "allowlist": 2}

Emit = Callable[[dict[str, Any]], None]


def permission_shortfall(needed: dict[str, Any], granted: dict[str, Any]) -> str | None:
    """Name the first permission a capability needs that the experiment does not grant."""
    if _FILESYSTEM[needed["filesystem"]] > _FILESYSTEM[granted["filesystem"]]:
        return f"filesystem '{needed['filesystem']}'"
    if _NETWORK[needed["network"]] > _NETWORK[granted["network"]]:
        return f"network '{needed['network']}'"
    for flag in ("subprocess", "gpu"):
        if needed.get(flag, False) and not granted.get(flag, False):
            return flag
    missing = set(needed.get("models", [])) - set(granted.get("models", []))
    if missing:
        return f"model '{sorted(missing)[0]}'"
    return None


def _first_error(schema: dict[str, Any], instance: Any) -> str | None:
    errors = sorted(
        Draft202012Validator(schema).iter_errors(instance),
        key=lambda e: [str(part) for part in e.absolute_path],
    )
    if not errors:
        return None
    where = "".join(f"/{part}" for part in errors[0].absolute_path) or "<root>"
    return f"{where}: {errors[0].message}"


class ToolHost:
    def __init__(
        self,
        capabilities: dict[str, tuple[Path, dict[str, Any]]],
        workspace: Path,
        permissions: dict[str, Any],
        emit: Emit,
        actor: dict[str, Any],
        allow: list[str] | None = None,
        deny: list[str] | None = None,
        registry: Registry | None = None,
    ):
        self.capabilities = capabilities
        self.workspace = workspace
        self.permissions = permissions
        self.emit = emit
        self.actor = actor
        self.allow = None if allow is None else set(allow)
        self.deny = set(deny or [])
        self._calls = 0
        self.registry = registry

    def call(self, capability_id: Any, arguments: Any) -> dict[str, Any]:
        """Run one tool call and return what the model is told about it."""
        if not (isinstance(capability_id, str) and _SLUG.match(capability_id)
                and len(capability_id) <= 128 and isinstance(arguments, dict)):
            # Not expressible as a tool.call event, so it is recorded as what it is.
            message = "malformed tool call: need a capability id and an argument object"
            self.emit({
                "recorder": "tool_host", "type": "error", "actor": self.actor,
                "payload": {"origin": "agent_runtime", "message": message, "recoverable": True},
            })
            return {"status": "error", "error": message}

        self._calls += 1
        correlation_id = f"call-{self._calls:06d}"
        known = self.capabilities.get(capability_id)
        call_payload: dict[str, Any] = {"capability_id": capability_id, "arguments": arguments}
        if known:
            call_payload["capability_version"] = known[1]["version"]
        self.emit({
            "recorder": "tool_host", "type": "tool.call", "actor": self.actor,
            "correlation_id": correlation_id, "payload": call_payload,
        })

        started = time.monotonic()
        outcome = (self._discovery(capability_id, arguments) if self.registry is not None
                   else self._run(capability_id, arguments))
        outcome["duration_ms"] = int((time.monotonic() - started) * 1000)
        self.emit({
            "recorder": "tool_host", "type": "tool.result", "actor": {"kind": "system"},
            "correlation_id": correlation_id, "payload": outcome,
        })
        return {key: value for key, value in outcome.items() if key != "duration_ms"}

    def _unavailable(self, capability_id: str) -> str | None:
        if capability_id in self.deny or (self.allow is not None
                                          and capability_id not in self.allow):
            return "not available in this experiment"
        needed = self.capabilities[capability_id][1]["permissions"]
        shortfall = permission_shortfall(needed, self.permissions)
        return f"needs permission not granted: {shortfall}" if shortfall else None

    def _discovery(self, name: str, arguments: dict) -> dict:
        if name not in META_SCHEMAS:
            return {"status": "error", "error": f"unknown discovery tool '{name}'"}
        problem = _first_error(META_SCHEMAS[name], arguments)
        if problem:
            return {"status": "error", "error": f"invalid_input: {problem}"}
        if name == "capabilities.search":
            available = {ident for ident in self.capabilities if not self._unavailable(ident)}
            return {"status": "ok", "result": {"matches": self.registry.search(
                arguments["query"], available, arguments.get("category"))}}
        ident = arguments["id"]
        if ident not in self.capabilities:
            return {"status": "error", "error": f"unknown capability '{ident}'"}
        denied = self._unavailable(ident)
        if denied:
            return {"status": "denied", "error": denied}
        manifest = self.capabilities[ident][1]
        if name == "capabilities.describe":
            return {"status": "ok", "result": {"id": ident,
                    "documentation": self.registry.describe(self.registry.entries[ident]),
                    "input_schema": manifest["input_schema"],
                    "example_call": {"tool": "capabilities.invoke", "arguments": {
                        "id": ident, "arguments": manifest["demonstrations"][0]["input"]}}}}
        if name == "capabilities.invoke":
            return self._run(ident, arguments["arguments"])
        demos = manifest["demonstrations"]
        demo_id = arguments.get("demo_id", demos[0]["id"])
        demo = next((d for d in demos if d["id"] == demo_id), None)
        if demo is None:
            return {"status": "error", "error": f"unknown demonstration '{demo_id}'"}
        outcome = self._run(ident, demo["input"])
        result = {"id": ident, "demo_id": demo_id, "input": demo["input"], "outcome": outcome}
        if "expected_output" in demo:
            result["expected_output"] = demo["expected_output"]
            result["matches_expected"] = (outcome.get("status") == "ok"
                                           and outcome.get("result") == demo["expected_output"])
        return {"status": "ok", "result": result}

    def _run(self, capability_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if capability_id not in self.capabilities:
            return {"status": "error", "error": f"unknown capability '{capability_id}'"}
        directory, manifest = self.capabilities[capability_id]
        denied = self._unavailable(capability_id)
        if denied:
            return {"status": "denied", "error": denied}
        problem = _first_error(manifest["input_schema"], arguments)
        if problem:
            return {"status": "error", "error": f"invalid_input: {problem}"}

        invocation = manifest["invocation"]
        if invocation["interface"] == "python":
            command = [sys.executable, "-I", "-B", str(_RUNNER), invocation["entrypoint"]]
        elif invocation["interface"] == "subprocess":
            command = list(invocation["command"])
            if command[0] == "python":
                command[0] = sys.executable
        else:
            return {
                "status": "error",
                "error": f"unsupported_interface: '{invocation['interface']}'",
            }

        extra = {}
        # The workspace is the only place a capability is ever told about.
        if manifest["permissions"]["filesystem"] != "none":
            extra["OPEN_LAB_WORKSPACE"] = str(self.workspace)
        timeout_s = manifest.get("resources", {}).get("timeout_s", DEFAULT_TIMEOUT_S)
        try:
            done = subprocess.run(  # noqa: S603 - argument vector, never a shell
                command,
                input=json.dumps(arguments).encode("utf-8"),
                capture_output=True,
                cwd=directory,
                env=minimal_environment(extra),
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "error": f"no result within {timeout_s} s"}
        except OSError as error:
            return {"status": "error", "error": f"execution_error: {error.strerror or error}"}

        try:
            payload = json.loads(done.stdout.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            payload = None
        if done.returncode != 0:
            if (isinstance(payload, dict) and set(payload) == {"error"}
                    and isinstance(payload["error"], str)):
                return {"status": "error", "error": f"capability_error: {payload['error']}"}
            lines = done.stderr.decode("utf-8", errors="replace").strip().splitlines()
            detail = lines[-1] if lines else "no error output"
            return {
                "status": "error",
                "error": f"execution_error: exit status {done.returncode}: {detail}",
            }
        if payload is None:
            return {"status": "error", "error": "invalid_output: output is not JSON"}
        problem = _first_error(manifest["output_schema"], payload)
        if problem:
            return {"status": "error", "error": f"invalid_output: {problem}"}
        return {"status": "ok", "result": payload}
