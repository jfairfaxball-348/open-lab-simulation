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

from . import faults, phenomena, run_time
from .agent_tools import CREATE, AgentTools
from .container import WORKSPACE_MOUNT, Backend, ContainerUnavailable, container_name
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
        admission=None,
        sandbox: Sandbox | None = None,
        before_execute=None,
        service=None,
        agent_tools_enabled=False,
        run_id=None,
        contracts=None,
        registrations=None,
        world_service=None,
        time_service=None,
        scheduled=False,
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
        self.admission = admission
        self.sandbox = sandbox
        self.before_execute = before_execute
        self.service = service
        self.world_service = world_service
        self.time_service, self.scheduled = time_service, scheduled
        self.sleep_requested = False
        self.agent_tools = None
        if agent_tools_enabled:
            if sandbox is None or contracts is None or not run_id:
                raise ValueError("agent tools require a configured container sandbox")
            self.agent_tools = AgentTools(self, run_id, contracts, registrations)
        elif registrations:
            raise ValueError("cannot restore disabled agent tools")

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
        denied = ("waiting for scheduled wake" if self.sleep_requested else
                  self.admission() if self.admission is not None else None)
        outcome = ({"status": "denied", "error": denied} if denied else
                   self._discovery(capability_id, arguments) if self.registry is not None
                   else self._run(capability_id, arguments))
        outcome["duration_ms"] = int((time.monotonic() - started) * 1000)
        self.emit({
            "recorder": "tool_host", "type": "tool.result", "actor": {"kind": "system"},
            "correlation_id": correlation_id, "payload": outcome,
        })
        return {key: value for key, value in outcome.items() if key != "duration_ms"}

    def _unavailable(self, capability_id: str) -> str | None:
        if capability_id in run_time.IDS and (self.time_service is None
                                              or capability_id == "time.sleep"
                                              and not self.scheduled):
            return "time service is not enabled by this runtime policy"
        if (capability_id in phenomena.IDS
                and self.capabilities[capability_id][1]["invocation"] == phenomena.invocation(
                    capability_id) and self.world_service is None):
            return "world phenomena is not configured"
        if capability_id == "tools.create" or (self.agent_tools and
                                             capability_id in self.agent_tools.records):
            if self.agent_tools is None or self.sandbox is None:
                return "agent-created tools are not enabled in this experiment"
            if "python.run" not in self.capabilities:
                return "agent tools require python.run"
            denied = self._unavailable("python.run")
            if denied:
                return f"python.run unavailable: {denied}"
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
        if capability_id in run_time.IDS and invocation == run_time.invocation(capability_id):
            if self.time_service is None:
                return {"status": "denied", "error": "time service unavailable"}
            if self.before_execute and self.before_execute(capability_id, arguments):
                return json.loads(json.dumps(faults.OUTCOME))
            outcome = self.time_service(capability_id, arguments)
            if outcome.get("status") == "ok":
                problem = _first_error(manifest["output_schema"], outcome.get("result"))
                if problem:
                    return {"status": "error", "error": f"invalid_output: {problem}"}
                if capability_id == "time.sleep":
                    self.sleep_requested = True
            return outcome
        if capability_id in phenomena.IDS and invocation == phenomena.invocation(capability_id):
            if self.world_service is None:
                return {"status": "denied", "error": "world phenomena is not configured"}
            if self.before_execute and self.before_execute(capability_id, arguments):
                return json.loads(json.dumps(faults.OUTCOME))
            outcome = self.world_service(capability_id, arguments)
            if outcome.get("status") == "ok":
                problem = _first_error(manifest["output_schema"], outcome.get("result"))
                if problem:
                    return {"status": "error", "error": f"invalid_output: {problem}"}
            return outcome
        if invocation == CREATE or (self.agent_tools and
                                    capability_id in self.agent_tools.records):
            if self.agent_tools is None or self.sandbox is None:
                return {"status": "denied", "error": "agent tools require the container tier"}
            if self.before_execute and self.before_execute(capability_id, arguments):
                return json.loads(json.dumps(faults.OUTCOME))
            try:
                return ({"status": "ok", "result": self.agent_tools.create(arguments)}
                        if invocation == CREATE else self.agent_tools.invoke(
                            capability_id, arguments))
            except OSError:
                return {"status": "error", "error": "agent_tool_error: cannot read tool source"}
            except (ValueError, RecursionError) as error:
                return {"status": "error", "error": f"agent_tool_error: {error}"}
        if invocation == {"interface": "mcp", "server": "supervisor-gateway",
                          "tool": "models.ask"}:
            if self.before_execute and self.before_execute(capability_id, arguments):
                return json.loads(json.dumps(faults.OUTCOME))
            if self.service is None:
                return {"status": "denied", "error": "supervisor service unavailable"}
            outcome = self.service(arguments)
            if outcome.get("status") == "ok":
                problem = _first_error(manifest["output_schema"], outcome.get("result"))
                if problem:
                    return {"status": "error", "error": f"invalid_output: {problem}"}
            return outcome
        # Inside a container the interpreter is the image's, and the runner is passed
        # as source so that no Simulation file is mounted into the sandbox.
        python = "python" if self.sandbox else sys.executable
        if invocation["interface"] == "python":
            runner = (["-c", _RUNNER.read_text(encoding="utf-8")] if self.sandbox
                      else [str(_RUNNER)])
            command = [python, "-I", "-B", *runner, invocation["entrypoint"]]
        elif invocation["interface"] == "subprocess":
            command = list(invocation["command"])
            if command[0] == "python":
                command[0] = python
        else:
            return {
                "status": "error",
                "error": f"unsupported_interface: '{invocation['interface']}'",
            }

        if self.before_execute and self.before_execute(capability_id, arguments):
            # Permission/schema/interface checks precede exposure; no code has run.
            return json.loads(json.dumps(faults.OUTCOME))
        extra = {}
        filesystem = manifest["permissions"]["filesystem"]
        # The workspace is the only place a capability is ever told about.
        if filesystem != "none":
            extra["OPEN_LAB_WORKSPACE"] = (WORKSPACE_MOUNT if self.sandbox
                                           else str(self.workspace))
        timeout_s = manifest.get("resources", {}).get("timeout_s", DEFAULT_TIMEOUT_S)
        stdin = json.dumps(arguments).encode("utf-8")
        if self.sandbox:
            done = self.sandbox.run(directory, self.workspace, filesystem, command, extra,
                                    stdin, timeout_s)
            if isinstance(done, str):
                return {"status": "error", "error": f"execution_error: {done}"}
            if done.timed_out:
                return {"status": "timeout", "error": f"no result within {timeout_s} s"}
        else:
            try:
                done = subprocess.run(  # noqa: S603 - argument vector, never a shell
                    command,
                    input=stdin,
                    capture_output=True,
                    cwd=directory,
                    env=minimal_environment(extra),
                    timeout=timeout_s,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return {"status": "timeout", "error": f"no result within {timeout_s} s"}
            except OSError as error:
                return {"status": "error",
                        "error": f"execution_error: {error.strerror or error}"}

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


class Sandbox:
    """Runs capability processes in containers for one run (the ``container`` tier)."""

    def __init__(self, backend: Backend, world: Path, run_id: str, limits: dict[str, Any]):
        self.backend = backend
        self.world = world
        self.run_id = run_id
        self.limits = limits

    def run(self, directory: Path, workspace: Path, filesystem: str, command: list[str],
            env: dict[str, str], stdin: bytes, timeout_s: int):
        """The finished process, or a message saying why no container could run."""
        name = container_name(self.run_id)
        environment = {"PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8", **env}
        try:
            vector = self.backend.command(
                name=name, run_id=self.run_id, world=self.world, capability_dir=directory,
                workspace=None if filesystem == "none" else workspace,
                writable=filesystem == "workspace_write", argv=command, env=environment,
                limits=self.limits, timeout_s=timeout_s,
            )
            return self.backend.run(vector, name, stdin, timeout_s, minimal_environment())
        except ContainerUnavailable as error:
            return f"container backend unavailable: {error}"
        except OSError as error:
            return f"wsl.exe: {error.strerror or error}"
