"""Experiment manager and supervisor: launch one run and stay in charge of it.

This process is the parent of the agent runtime. It alone holds the run
archive and the recorder, enforces the deadline, and decides how a run ends.
It never reads the agent's output to make a decision: a limit is a limit
whatever the agent says.
"""

from __future__ import annotations

import json
import queue
import re
import secrets
import shutil
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import __version__, ollama
from .artifacts import WorkspaceScanner
from .model import SCRIPTED_MODEL_ID, is_loopback
from .paths import data_root, minimal_environment
from .proctree import ProcessTree
from .recorder import Recorder, RecordError, utc_now
from .world import World, WorldError, compute_content_digest, load_world

_SLUG = re.compile(r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$")
_SEMVER = re.compile(r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$")

# What the runtime may ask to have recorded, and under which recorder name.
# Everything else (state changes, enforcement, artifacts) is the supervisor's
# alone, so the runtime cannot write itself a different history.
_RUNTIME_EVENTS = {
    "model.call": {"model_gateway"},
    "agent.output": {"agent_runtime"},
    "tool.call": {"tool_host"},
    "tool.result": {"tool_host"},
    "error": {"agent_runtime", "tool_host"},
}
_REQUEST_FIELDS = {"recorder", "type", "actor", "payload", "correlation_id"}
# -I: the runtime ignores PYTHON* variables and the user site directory.
WORKER_COMMAND = [sys.executable, "-I", "-m", "open_lab_simulation.worker"]


class LaunchError(Exception):
    """The experiment was refused before anything was created."""


@dataclass
class RunResult:
    run_id: str
    archive: Path
    workspace: Path
    state: str
    reason: str
    events: int
    notes: list[str] = field(default_factory=list)


def _unsupported(config: dict[str, Any], world: World) -> str | None:
    """Things the contracts allow but this runtime cannot yet honour. Refuse, never pretend."""
    if config["termination"]["on_deadline"] != "hard_stop" or "checkpointing" in config:
        return "checkpointing is not implemented; use termination.on_deadline 'hard_stop'"
    if "notify_agent_before_s" in config["termination"]:
        return "deadline notices are not implemented"
    if config["permissions"].get("subprocess") or config.get("capabilities", {}).get(
        "agent_created_tools"
    ):
        return "running agent-supplied code needs the container isolation tier"
    if config["permissions"]["network"] == "allowlist":
        return "network access cannot be granted: this runtime cannot enforce an allowlist"
    if world.manifest["environment"]["isolation"]["minimum_tier"] != "process":
        return "the world needs the container isolation tier, which is not available"
    return None


def _resolve_model(config: dict[str, Any], world: World) -> dict[str, Any]:
    """Decide exactly which model drives the run, or refuse.

    The experiment names a gateway model id; the world's model-provider
    manifests say what serves it. The server is asked what it actually holds,
    so the archive can name the model by digest.
    """
    orchestrator = config["orchestrator"]
    model_id = orchestrator["model_id"]
    if model_id == SCRIPTED_MODEL_ID:
        extension = config.get("extensions", {}).get("scripted_model")
        if not isinstance(extension, dict) or not isinstance(extension.get("replies"), list):
            raise LaunchError("the scripted model needs extensions.scripted_model.replies")
        return {"kind": "scripted", "model_id": SCRIPTED_MODEL_ID, "real_model": False,
                "replies": extension["replies"]}

    wanted_provider = orchestrator.get("provider_id")
    found = [
        (provider_id, document["provider"], model)
        for provider_id, document in world.providers.items()
        if wanted_provider in (None, provider_id)
        for model in document["models"] if model["id"] == model_id
    ]
    if len(found) != 1:
        raise LaunchError(f"the world does not define exactly one model '{model_id}'")
    provider_id, provider, model = found[0]
    api = provider["api"]
    if provider["kind"] != "ollama" or api["protocol"] != "openai_chat_completions":
        raise LaunchError("only Ollama providers with the chat-completions API are supported")
    if provider["locality"] != "local" or not is_loopback(api.get("base_url", "")):
        raise LaunchError("the model server must be on this machine")
    if "orchestrator" not in model["roles"]:
        raise LaunchError(f"model '{model_id}' is not an orchestrator")
    # Authority comes from the experiment: it must grant the model and the loopback call.
    permissions = config["permissions"]
    if permissions["network"] != "loopback" or model_id not in permissions.get("models", []):
        raise LaunchError(
            f"the experiment must grant network 'loopback' and list '{model_id}' in "
            "permissions.models"
        )
    try:
        identity = ollama.identify(ollama.native_base(api["base_url"]), model["provider_ref"])
    except ollama.OllamaError as error:
        raise LaunchError(str(error)) from None
    resolved = {
        "kind": "openai_chat", "model_id": model_id, "provider_id": provider_id,
        "real_model": True, "base_url": api["base_url"], "provider_ref": model["provider_ref"],
        "sampling": orchestrator.get("sampling", {}), "identity": identity,
    }
    if "context_window_tokens" in model:
        resolved["context_tokens"] = model["context_window_tokens"]
    return resolved


def _prepare(
    config_path: Path, root: Path
) -> tuple[bytes, dict[str, Any], World, dict[str, Any]]:
    try:
        raw = config_path.read_bytes()
        config = json.loads(raw.decode("utf-8"))
        named = config["world"]
        world_id, version = named["world_id"], named["version"]
    except (OSError, ValueError, KeyError, TypeError):
        raise LaunchError("the experiment configuration cannot be read") from None
    # Checked before use in a path: the schema that would reject them lives in the world.
    if not (isinstance(world_id, str) and _SLUG.match(world_id)
            and isinstance(version, str) and _SEMVER.match(version)):
        raise LaunchError("the experiment names no valid world")
    expected = named.get("content_digest") if isinstance(named, dict) else None
    try:
        world = load_world(root / "worlds" / world_id / version, expected)
    except WorldError as error:
        raise LaunchError(f"world {world_id} {version} refused: {error}") from None
    problems = world.contracts.validate(config, "experiment-config")
    if problems:
        raise LaunchError(f"invalid experiment configuration: {problems[0]}")
    reason = _unsupported(config, world)
    if reason:
        raise LaunchError(f"not supported by this runtime: {reason}")
    return raw, config, world, _resolve_model(config, world)


def _new_run_id() -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dt%H%M%Sz")
    return f"run-{stamp}-{secrets.token_hex(3)}"


def run_experiment(config_path: Path, root: Path | None = None) -> RunResult:
    """Launch one run and supervise it to its end. Raises ``LaunchError`` if refused."""
    root = root or data_root()
    raw, config, world, model = _prepare(config_path, root)
    experiment_id = config["experiment_id"]
    run_id = _new_run_id()
    archive = root / "runs" / experiment_id / run_id
    workspace = root / "workspaces" / run_id
    # Neither may exist: two runs never share an archive or a workspace.
    archive.mkdir(parents=True)
    workspace.mkdir(parents=True)
    (archive / "experiment.json").write_bytes(raw)
    shutil.copyfile(world.directory / "world-release.json", archive / "world-release.json")
    shutil.copytree(world.directory / "contracts", archive / "contracts")
    # Exactly which model ran, as the server reported it at launch.
    described = {key: value for key, value in model.items() if key != "replies"}
    (archive / "model.json").write_text(
        json.dumps(described, indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    recorder = Recorder(archive / "events.jsonl", world.contracts, experiment_id, run_id)
    system = {"kind": "system"}
    supervisor = {"kind": "supervisor"}
    agent = {"kind": "agent", "model_id": model["model_id"]}
    if "provider_id" in model:
        agent["provider_id"] = model["provider_id"]
    scanner = WorkspaceScanner(workspace, archive, recorder, world.contracts, agent)
    started_at = utc_now()
    state = "created"

    def change(to: str, reason: str, by: str = "supervisor") -> None:
        nonlocal state
        recorder.record(
            recorder=by, type="run.state_changed",
            actor=system if by == "experiment_manager" else supervisor,
            payload={"from": state, "to": to, "reason": reason},
        )
        state = to

    recorder.record(recorder="experiment_manager", type="run.state_changed", actor=system,
                    payload={"to": "created", "reason": "launch"})
    change("starting", "launch", by="experiment_manager")

    notes: list[str] = []
    tree: ProcessTree | None = None
    stderr_log = (archive / "runtime.stderr.log").open("wb")
    final, reason = "failed", "supervisor error"
    try:
        tree = ProcessTree(
            WORKER_COMMAND,
            stdin=-1, stdout=-1, stderr=stderr_log, env=minimal_environment(),
        )
        capabilities = config.get("capabilities", {})
        spec = {
            "release_dir": str(world.directory),
            "workspace": str(workspace),
            "capabilities": world.manifest["capabilities"],
            "initial_prompt": config["initial_prompt"],
            "max_steps": config["limits"].get("max_steps"),
            "permissions": config["permissions"],
            "allow": capabilities.get("allow"),
            "deny": capabilities.get("deny"),
            "model": model,
            "actor": agent,
        }
        process = tree.process
        process.stdin.write((json.dumps(spec) + "\n").encode("utf-8"))
        process.stdin.flush()
        change("running", "runtime started")
        final, reason = _supervise(
            tree, recorder, scanner, config["limits"]["wall_clock_s"], supervisor
        )
        if final == "terminated":
            change("stopping", reason)
    except (OSError, RecordError, ValueError) as error:
        notes.append(f"supervisor error: {type(error).__name__}: {error}")
    finally:
        if tree is not None:
            tree.close()
        stderr_log.close()

    # From here nothing of the run is alive. Finish the record.
    try:
        scanner.scan({"kind": "system"})
        if compute_content_digest(world.directory, world.manifest) != world.digest:
            recorder.record(
                recorder="supervisor", type="error", actor=supervisor,
                payload={"origin": "host", "recoverable": False,
                         "message": "the world release changed while the run was in progress"},
            )
            notes.append("the world release changed during the run")
        if notes and final != "failed":
            final, reason = "failed", notes[0]
        if final == "failed" and not recorder.by_type.get("error"):
            recorder.record(
                recorder="supervisor", type="error", actor=supervisor,
                payload={"origin": "supervisor", "message": reason, "recoverable": False},
            )
        change(final, reason)
    finally:
        recorder.close()

    summary = {
        "experiment_id": experiment_id,
        "run_id": run_id,
        "final_state": final,
        "reason": reason,
        "started_at": started_at,
        "ended_at": utc_now(),
        "event_count": recorder.count,
        "events_by_type": dict(sorted(recorder.by_type.items())),
        "artifact_versions": scanner.versions,
        "last_event_sha256": recorder.last_event_sha256,
        "world": {"world_id": world.manifest["world_id"], "version": world.manifest["version"],
                  "content_digest": world.digest},
        "model": {
            "model_id": model["model_id"], "real_model": model["real_model"],
            **({"provider_id": model["provider_id"], "provider_ref": model["provider_ref"],
                "digest": model["identity"].get("digest")} if model["real_model"] else {}),
        },
        "simulation_version": __version__,
    }
    (archive / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return RunResult(run_id, archive, workspace, final, reason, recorder.count, notes)


def _supervise(
    tree: ProcessTree,
    recorder: Recorder,
    scanner: WorkspaceScanner,
    wall_clock_s: int,
    supervisor: dict[str, Any],
) -> tuple[str, str]:
    """Relay the runtime's observations into the record until it ends or is stopped."""
    process = tree.process
    messages: queue.Queue[bytes | None] = queue.Queue()

    def pump() -> None:
        for line in process.stdout:
            messages.put(line)
        messages.put(None)

    threading.Thread(target=pump, daemon=True).start()
    deadline = time.monotonic() + wall_clock_s
    open_calls: dict[str, str] = {}  # correlation id -> capability id
    ended: dict[str, Any] | None = None

    def refuse(message: str) -> None:
        recorder.record(
            recorder="supervisor", type="error", actor=supervisor,
            payload={"origin": "agent_runtime", "recoverable": True,
                     "message": f"runtime request refused: {message}"},
        )

    def handle(line: bytes) -> None:
        nonlocal ended
        try:
            message = json.loads(line.decode("utf-8"))
        except ValueError:
            return refuse("not JSON")
        if isinstance(message, dict) and isinstance(message.get("end"), dict):
            ended = message["end"]
            return None
        request = message.get("event") if isinstance(message, dict) else None
        if not isinstance(request, dict) or not set(request) <= _REQUEST_FIELDS:
            return refuse("not an event request")
        kind = request.get("type")
        if request.get("recorder") not in _RUNTIME_EVENTS.get(kind, ()):
            return refuse(f"'{kind}' is not the runtime's to record")
        actor = request.get("actor")
        if not isinstance(actor, dict) or actor.get("kind") not in ("agent", "system"):
            return refuse("actor must be the agent or the system")
        try:
            event = recorder.record(
                recorder=request["recorder"], type=kind, actor=actor,
                payload=request.get("payload"), correlation_id=request.get("correlation_id"),
            )
        except RecordError as error:
            return refuse(f"invalid {kind} event: {error}")
        if kind == "tool.call":
            open_calls[event["correlation_id"]] = event["payload"]["capability_id"]
        elif kind == "tool.result":
            # The runtime is waiting for the acknowledgement, so whatever changed
            # in the workspace since the last scan happened during this call.
            capability_id = open_calls.pop(event["correlation_id"], None)
            producer: dict[str, Any] = {"kind": "tool",
                                        "tool_call_correlation_id": event["correlation_id"]}
            if capability_id:
                producer["capability_id"] = capability_id
            scanner.scan(producer)
        return None

    def acknowledge() -> None:
        try:
            process.stdin.write(b'{"ack":true}\n')
            process.stdin.flush()
        except OSError:
            pass  # the runtime has already gone; its exit is handled below

    while ended is None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            # Kill first, then write down what had already happened, then the action.
            tree.kill()
            _drain(messages, handle)
            recorder.record(
                recorder="supervisor", type="supervisor.action", actor=supervisor,
                payload={"action": "terminated", "detail": "wall_clock_s limit reached"},
            )
            return "terminated", "deadline"
        try:
            line = messages.get(timeout=min(remaining, 0.25))
        except queue.Empty:
            continue
        if line is None:
            code = process.wait()
            return "failed", f"the agent runtime exited unexpectedly (status {code})"
        handle(line)
        if ended is None:
            acknowledge()

    if ended.get("reason") == "max_steps":
        recorder.record(
            recorder="supervisor", type="supervisor.action", actor=supervisor,
            payload={"action": "limit_enforced", "detail": "max_steps limit reached"},
        )
        return "terminated", "max_steps"
    if ended.get("reason") == "model_error":
        return "failed", "the model server could not produce a reply"
    if ended.get("reason") == "no_tool_calls":
        return "completed", "the model made no tool call"
    return "completed", "the agent loop ended by itself"


def _drain(messages: queue.Queue[bytes | None], handle) -> None:
    """Record requests the runtime had already sent before it was killed."""
    while True:
        try:
            line = messages.get(timeout=2)
        except queue.Empty:
            return
        if line is None:
            return
        handle(line)
