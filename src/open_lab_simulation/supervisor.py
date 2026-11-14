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
from .archive import verify_archive
from .artifacts import WorkspaceScanner
from .checkpoints import (
    CheckpointError,
    CheckpointTimeout,
    latest_checkpoint,
    restore_workspace,
    run_lock,
    write_checkpoint,
)
from .model import SCRIPTED_MODEL_ID, is_loopback
from .paths import data_root, minimal_environment
from .proctree import ProcessTree
from .recorder import Recorder, RecordError, utc_now
from .registry import Registry, RegistryError
from .world import World, WorldError, compute_content_digest, load_world

_SLUG = re.compile(r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$")
_SEMVER = re.compile(r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$")

# What the runtime may ask to have recorded, and under which recorder name.
# Everything else (state changes, enforcement, artifacts) is the supervisor's
# alone, so the runtime cannot write itself a different history.
_RUNTIME_EVENTS = {
    "model.call": {"model_gateway"},
    "agent.output": {"agent_runtime"},
    # The continuation message: what the runtime itself said to the model.
    "environment.message": {"agent_runtime"},
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
    if (
        config["termination"]["on_deadline"] == "checkpoint_then_stop"
        and "checkpointing" not in config
    ):
        return "checkpoint_then_stop requires checkpointing"
    if "notify_agent_before_s" in config["termination"]:
        return "deadline notices are not implemented"
    if continuation_message(config, world) is None:
        return (
            "the world's contracts predate the continuation message (observation-event "
            "1.1.0, experiment-config 1.1.0); rebuild the world with current contracts"
        )
    kinds = world.contracts.schemas["observation-event"]["allOf"]
    if not any(
        "preamble"
        in rule.get("then", {})
        .get("properties", {})
        .get("payload", {})
        .get("properties", {})
        .get("kind", {})
        .get("enum", [])
        for rule in kinds
    ):
        return "the world's contracts cannot record the discovery preamble; rebuild the world"
    if config["permissions"].get("subprocess") or config.get("capabilities", {}).get(
        "agent_created_tools"
    ):
        return "running agent-supplied code needs the container isolation tier"
    if config["permissions"]["network"] == "allowlist":
        return "network access cannot be granted: this runtime cannot enforce an allowlist"
    if world.manifest["environment"]["isolation"]["minimum_tier"] != "process":
        return "the world needs the container isolation tier, which is not available"
    return None


def continuation_message(config: dict[str, Any], world: World) -> str | None:
    """The text given to the model when it replies without calling a tool.

    The experiment's own text, or the default stated in the world's bundled
    experiment-config schema, so the Simulation holds no second copy of it.
    ``None`` if the world's contracts cannot express or record a continuation.
    """
    try:
        events = world.contracts.schemas["observation-event"]
        if "environment.message" not in events["properties"]["type"]["enum"]:
            return None
        default = world.contracts.schemas["experiment-config"]["properties"]["continuation"][
            "properties"
        ]["message"]["default"]
    except KeyError:
        return None
    message = config.get("continuation", {}).get("message", default)
    return message if isinstance(message, str) and message else None


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
        return {
            "kind": "scripted",
            "model_id": SCRIPTED_MODEL_ID,
            "real_model": False,
            "replies": extension["replies"],
        }

    wanted_provider = orchestrator.get("provider_id")
    found = [
        (provider_id, document["provider"], model)
        for provider_id, document in world.providers.items()
        if wanted_provider in (None, provider_id)
        for model in document["models"]
        if model["id"] == model_id
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
        "kind": "openai_chat",
        "model_id": model_id,
        "provider_id": provider_id,
        "real_model": True,
        "base_url": api["base_url"],
        "provider_ref": model["provider_ref"],
        "sampling": orchestrator.get("sampling", {}),
        "identity": identity,
    }
    if "context_window_tokens" in model:
        resolved["context_tokens"] = model["context_window_tokens"]
    return resolved


def _prepare(config_path: Path, root: Path) -> tuple[bytes, dict[str, Any], World, dict[str, Any]]:
    try:
        raw = config_path.read_bytes()
        config = json.loads(raw.decode("utf-8"))
        named = config["world"]
        world_id, version = named["world_id"], named["version"]
    except (OSError, ValueError, KeyError, TypeError):
        raise LaunchError("the experiment configuration cannot be read") from None
    # Checked before use in a path: the schema that would reject them lives in the world.
    if not (
        isinstance(world_id, str)
        and _SLUG.match(world_id)
        and isinstance(version, str)
        and _SEMVER.match(version)
    ):
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
    try:
        Registry(world.directory, world.capabilities)
    except RegistryError as error:
        raise LaunchError(str(error)) from None
    return raw, config, world, _resolve_model(config, world)


def _new_run_id() -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dt%H%M%Sz")
    return f"run-{stamp}-{secrets.token_hex(3)}"


def run_experiment(
    config_path: Path, root: Path | None = None, *, pause_after_steps: int | None = None
) -> RunResult:
    """Launch one fresh run. Operator pauses require configured checkpointing."""
    root = (root or data_root()).resolve()
    raw, config, world, model = _prepare(config_path, root)
    if pause_after_steps is not None and (pause_after_steps < 1 or "checkpointing" not in config):
        raise LaunchError("pausing needs checkpointing and a positive pause-after-steps value")
    run_id = _new_run_id()
    archive = root / "runs" / config["experiment_id"] / run_id
    workspace = root / "workspaces" / run_id
    archive.mkdir(parents=True)
    workspace.mkdir(parents=True)
    with run_lock(archive):
        return _run_prepared(
            raw, config, world, model, archive, workspace, pause_after_steps=pause_after_steps
        )


def resume_experiment(
    archive: Path, root: Path | None = None, *, pause_after_steps: int | None = None
) -> RunResult:
    """Append a segment to the same run; never remove or rewrite observations."""
    root = (root or data_root()).resolve()
    archive = archive.resolve()
    if (
        archive.parent.parent != root / "runs"
        or not _SLUG.fullmatch(archive.name)
        or not _SLUG.fullmatch(archive.parent.name)
    ):
        raise LaunchError("resume needs an archive beneath this data root's runs directory")
    try:
        with run_lock(archive):
            problems = verify_archive(archive, checkpoints=False)
            if problems:
                raise LaunchError(f"run archive cannot be resumed: {problems[0]}")
            summary = json.loads((archive / "summary.json").read_text(encoding="utf-8"))
            if summary["final_state"] not in ("paused", "terminated", "failed"):
                raise LaunchError("only a stopped, incomplete run can be resumed")
            raw, config, world, model = _prepare(archive / "experiment.json", root)
            if "checkpointing" not in config:
                raise LaunchError("run did not enable checkpointing")
            if world.digest != summary["world"]["content_digest"]:
                raise LaunchError("resuming requires the original world digest")
            prior_model = json.loads((archive / "model.json").read_text(encoding="utf-8"))
            if model.get("identity", {}).get("digest") != prior_model.get("identity", {}).get(
                "digest"
            ):
                raise LaunchError("resuming requires the original model digest")
            events = [
                json.loads(line)
                for line in (archive / "events.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            metadata, state, rejected = latest_checkpoint(archive, events, world.contracts)
            if (
                metadata["extensions"]["world_digest"] != world.digest
                or metadata["orchestrator_model_id"] != model["model_id"]
            ):
                raise LaunchError("checkpoint world or model identity differs")
            elapsed = max(
                summary.get("elapsed_s", 0),
                metadata["progress"]["elapsed_s"],
                events[-1].get("monotonic_ms", 0) / 1000,
            )
            consumed = sum(e["type"] == "model.call" for e in events)
            if elapsed >= config["limits"]["wall_clock_s"] or consumed >= config["limits"].get(
                "max_steps", float("inf")
            ):
                raise LaunchError("run has exhausted its original time or step budget")
            if pause_after_steps is not None and pause_after_steps <= state["steps"]:
                raise LaunchError("pause-after-steps must exceed the restored step count")
            state["call_count"] = max(
                state["call_count"],
                max(
                    (
                        int(e["correlation_id"].split("-")[-1])
                        for e in events
                        if e["type"] == "tool.call"
                        and re.fullmatch(r"call-[0-9]+", e["correlation_id"])
                    ),
                    default=0,
                ),
            )
            workspace = root / "workspaces" / archive.name
            restore_workspace(archive, metadata, state, workspace)
            resume = {
                "metadata": metadata,
                "state": state,
                "rejected": rejected,
                "elapsed": elapsed,
                "consumed": consumed,
                "summary": summary,
            }
            return _run_prepared(
                raw,
                config,
                world,
                model,
                archive,
                workspace,
                resume=resume,
                pause_after_steps=pause_after_steps,
            )
    except (CheckpointError, OSError, ValueError, KeyError) as error:
        raise LaunchError(f"resume refused: {error}") from None


def _run_prepared(
    raw, config, world, model, archive, workspace, *, resume=None, pause_after_steps=None
):
    run_id, experiment_id = archive.name, config["experiment_id"]
    if resume is None:
        (archive / "experiment.json").write_bytes(raw)
        shutil.copyfile(world.directory / "world-release.json", archive / "world-release.json")
        shutil.copytree(world.directory / "contracts", archive / "contracts")
        described = {key: value for key, value in model.items() if key != "replies"}
        (archive / "model.json").write_text(
            json.dumps(described, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
    elapsed_offset = resume["elapsed"] if resume else 0
    segment_started = time.monotonic()
    recorder = Recorder(
        archive / "events.jsonl",
        world.contracts,
        experiment_id,
        run_id,
        append=resume is not None,
        elapsed_s=elapsed_offset,
    )
    system = {"kind": "system"}
    supervisor = {"kind": "supervisor"}
    agent = {"kind": "agent", "model_id": model["model_id"]}
    if "provider_id" in model:
        agent["provider_id"] = model["provider_id"]
    scanner = WorkspaceScanner(
        workspace, archive, recorder, world.contracts, agent, resume=resume is not None
    )
    started_at = resume["summary"]["started_at"] if resume else utc_now()
    state = resume["summary"]["final_state"] if resume else "created"

    def change(to: str, reason: str, by: str = "supervisor") -> None:
        nonlocal state
        recorder.record(
            recorder=by,
            type="run.state_changed",
            actor=system if by == "experiment_manager" else supervisor,
            payload={"from": state, "to": to, "reason": reason},
        )
        state = to

    if resume:
        segments = archive / "segments"
        segments.mkdir(exist_ok=True)
        shutil.copyfile(archive / "summary.json", segments / f"{recorder.count:08d}.json")
        (archive / "summary.json").unlink()
        change("recovering", "operator requested checkpoint resume")
        recorder.record(
            recorder="supervisor",
            type="recovery",
            actor=supervisor,
            payload={
                "action": "resume_from_checkpoint",
                "checkpoint_id": resume["metadata"]["checkpoint_id"],
                "extensions": {
                    "snapshot_event_seq": resume["metadata"]["last_event_seq"],
                    "rejected_checkpoints": resume["rejected"],
                    "elapsed_s": elapsed_offset,
                },
            },
        )
        scanner.scan({"kind": "system"})
    else:
        recorder.record(
            recorder="experiment_manager",
            type="run.state_changed",
            actor=system,
            payload={"to": "created", "reason": "launch"},
        )
        change("starting", "launch", by="experiment_manager")

    notes: list[str] = []
    tree: ProcessTree | None = None
    stderr_log = (archive / "runtime.stderr.log").open("ab" if resume else "wb")
    final, reason = "failed", "supervisor error"
    try:
        tree = ProcessTree(
            WORKER_COMMAND,
            stdin=-1,
            stdout=-1,
            stderr=stderr_log,
            env=minimal_environment(),
        )
        capabilities = config.get("capabilities", {})
        spec = {
            "release_dir": str(world.directory),
            "workspace": str(workspace),
            "capabilities": world.manifest["capabilities"],
            "initial_prompt": config["initial_prompt"],
            "continuation_message": continuation_message(config, world),
            "max_steps": (
                resume["state"]["steps"] + config["limits"]["max_steps"] - resume["consumed"]
                if resume and "max_steps" in config["limits"]
                else config["limits"].get("max_steps")
            ),
            "checkpointing": config.get("checkpointing"),
            "permissions": config["permissions"],
            "allow": capabilities.get("allow"),
            "deny": capabilities.get("deny"),
            "model": model,
            "actor": agent,
        }
        if resume:
            spec["runtime_state"] = resume["state"]
        process = tree.process
        process.stdin.write((json.dumps(spec) + "\n").encode("utf-8"))
        process.stdin.flush()
        change("running", "runtime started")

        def snapshot(runtime, reason):
            return write_checkpoint(
                archive,
                workspace,
                recorder,
                world,
                model["model_id"],
                runtime,
                elapsed_offset + time.monotonic() - segment_started,
                reason,
                config.get("checkpointing", {}).get("keep_last", 3),
                deadline=(
                    segment_started + config["limits"]["wall_clock_s"] - elapsed_offset
                    + (
                        config["termination"].get("grace_period_s", 0)
                        if config["termination"]["on_deadline"] == "checkpoint_then_stop"
                        else 0
                    )
                ),
            )

        final, reason = _supervise(
            tree,
            recorder,
            scanner,
            config["limits"]["wall_clock_s"]
            - (elapsed_offset + time.monotonic() - segment_started),
            supervisor,
            checkpointing=config.get("checkpointing"),
            snapshot=snapshot,
            pause_after_steps=pause_after_steps,
            termination=config["termination"],
        )
        if final == "terminated":
            change("stopping", reason)
    except CheckpointTimeout:
        tree.kill()
        recorder.record(
            recorder="supervisor",
            type="supervisor.action",
            actor=supervisor,
            payload={
                "action": "terminated",
                "detail": "deadline allowance exhausted during checkpoint creation",
            },
        )
        final, reason = "terminated", "deadline"
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
                recorder="supervisor",
                type="error",
                actor=supervisor,
                payload={
                    "origin": "host",
                    "recoverable": False,
                    "message": "the world release changed while the run was in progress",
                },
            )
            notes.append("the world release changed during the run")
        if notes and final != "failed":
            final, reason = "failed", notes[0]
        if final == "failed" and not recorder.by_type.get("error"):
            recorder.record(
                recorder="supervisor",
                type="error",
                actor=supervisor,
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
        "elapsed_s": elapsed_offset + time.monotonic() - segment_started,
        "event_count": recorder.count,
        "events_by_type": dict(sorted(recorder.by_type.items())),
        "artifact_versions": scanner.versions,
        "last_event_sha256": recorder.last_event_sha256,
        "world": {
            "world_id": world.manifest["world_id"],
            "version": world.manifest["version"],
            "content_digest": world.digest,
        },
        "model": {
            "model_id": model["model_id"],
            "real_model": model["real_model"],
            **(
                {
                    "provider_id": model["provider_id"],
                    "provider_ref": model["provider_ref"],
                    "digest": model["identity"].get("digest"),
                }
                if model["real_model"]
                else {}
            ),
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
    *,
    checkpointing=None,
    snapshot=None,
    pause_after_steps=None,
    termination=None,
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
    termination = termination or {"on_deadline": "hard_stop"}
    graceful = termination["on_deadline"] == "checkpoint_then_stop"
    hard_deadline = deadline + (termination.get("grace_period_s", 0) if graceful else 0)
    deadline_requested = False
    deadline_checkpointed = False
    open_calls: dict[str, str] = {}  # correlation id -> capability id
    ended: dict[str, Any] | None = None
    last_seq: int | None = None
    next_checkpoint = time.monotonic()
    pause_requested = False
    stop_sent = False

    def refuse(message: str) -> None:
        recorder.record(
            recorder="supervisor",
            type="error",
            actor=supervisor,
            payload={
                "origin": "agent_runtime",
                "recoverable": True,
                "message": f"runtime request refused: {message}",
            },
        )

    def request_deadline() -> None:
        nonlocal deadline_requested
        if graceful and not deadline_requested and time.monotonic() >= deadline:
            deadline_requested = True
            recorder.record(
                recorder="supervisor",
                type="supervisor.action",
                actor=supervisor,
                payload={
                    "action": "limit_enforced",
                    "detail": "deadline reached; stop at the next complete-step checkpoint",
                    "extensions": {"stage": "deadline_checkpoint_requested"},
                },
            )

    def handle(line: bytes, *, stopped: bool = False) -> None:
        nonlocal ended, last_seq, next_checkpoint, stop_sent, deadline_checkpointed
        last_seq = None
        try:
            message = json.loads(line.decode("utf-8"))
        except ValueError:
            return refuse("not JSON")
        if isinstance(message, dict) and isinstance(message.get("state"), dict):
            # After killing a busy tree its live files no longer match a queued
            # barrier. Retain previous safe snapshots; never snapshot that state.
            if stopped:
                return None
            request_deadline()
            runtime = message["state"]
            fields = ("steps", "script_cursor", "call_count", "tokens_used")
            if (
                checkpointing is None
                or open_calls
                or not isinstance(runtime.get("messages"), list)
                or any(not isinstance(runtime.get(k), int) or runtime[k] < 0 for k in fields)
            ):
                return refuse("invalid checkpoint state barrier")
            stop_sent = deadline_requested or pause_requested or (
                pause_after_steps is not None and runtime["steps"] >= pause_after_steps
            )
            now = time.monotonic()
            if stop_sent or message.get("final") or now >= next_checkpoint:
                reason = (
                    "pre_shutdown"
                    if deadline_requested
                    else "operator_request"
                    if stop_sent
                    else "pre_shutdown"
                    if message.get("final")
                    else "interval"
                )
                try:
                    snapshot(runtime, reason)
                finally:
                    # I/O can cross the nominal deadline, including when the
                    # snapshot times out. Record enforcement before termination.
                    request_deadline()
                deadline_checkpointed = deadline_requested
                next_checkpoint = time.monotonic() + checkpointing.get("interval_s", 60)
            return None
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
                recorder=request["recorder"],
                type=kind,
                actor=actor,
                payload=request.get("payload"),
                correlation_id=request.get("correlation_id"),
            )
        except RecordError as error:
            return refuse(f"invalid {kind} event: {error}")
        last_seq = event["seq"]
        if kind == "tool.call":
            payload = event["payload"]
            target = payload["capability_id"]
            if target in ("capabilities.invoke", "capabilities.demonstrate"):
                candidate = payload.get("arguments", {}).get("id")
                if (
                    isinstance(candidate, str)
                    and _SLUG.fullmatch(candidate)
                    and len(candidate) <= 128
                ):
                    target = candidate
            open_calls[event["correlation_id"]] = target
        elif kind == "tool.result":
            # The runtime is waiting for the acknowledgement, so whatever changed
            # in the workspace since the last scan happened during this call.
            capability_id = open_calls.pop(event["correlation_id"], None)
            producer: dict[str, Any] = {
                "kind": "tool",
                "tool_call_correlation_id": event["correlation_id"],
            }
            if capability_id:
                producer["capability_id"] = capability_id
            scanner.scan(producer)
            if capability_id == "memory.write":
                outcome = event["payload"]
                result = outcome.get("result", {})
                if "outcome" in result:
                    outcome = result["outcome"]
                    result = outcome.get("result", {})
                if outcome.get("status") == "ok" and "key" in result:
                    recorder.record(
                        recorder="supervisor",
                        type="memory.written",
                        actor=supervisor,
                        payload={
                            "extensions": {
                                "key": result["key"],
                                "revision": result["revision"],
                                "tool_call_correlation_id": event["correlation_id"],
                            }
                        },
                    )
        return None

    def acknowledge() -> None:
        try:
            process.stdin.write(
                (json.dumps({"ack": True, "seq": last_seq, "stop": stop_sent}) + "\n").encode()
            )
            process.stdin.flush()
        except OSError:
            pass  # the runtime has already gone; its exit is handled below

    def deadline_stop(pending: bytes | None = None) -> tuple[str, str]:
        # Preserve observation order, but ignore state barriers after killing.
        tree.kill()
        if pending is not None:
            handle(pending, stopped=True)
        _drain(messages, lambda line: handle(line, stopped=True))
        recorder.record(
            recorder="supervisor",
            type="supervisor.action",
            actor=supervisor,
            payload={
                "action": "terminated",
                "detail": (
                    "deadline grace exhausted; retaining the previous safe checkpoint"
                    if graceful else "wall_clock_s limit reached"
                ),
            },
        )
        return "terminated", "deadline"

    while ended is None:
        request_deadline()
        remaining = hard_deadline - time.monotonic()
        if remaining <= 0:
            return deadline_stop()
        try:
            # Wake at the nominal deadline even when there is a grace allowance.
            wait = min(remaining, 0.25)
            if not deadline_requested and graceful:
                wait = min(wait, max(0.001, deadline - time.monotonic()))
            line = messages.get(timeout=wait)
        except queue.Empty:
            continue
        except KeyboardInterrupt:
            if checkpointing is None:
                tree.kill()
                return "terminated", "operator interrupt without checkpointing"
            pause_requested = True
            continue
        if line is None:
            code = process.wait()
            return "failed", f"the agent runtime exited unexpectedly (status {code})"
        if time.monotonic() >= hard_deadline:
            return deadline_stop(line)
        handle(line)
        if deadline_checkpointed:
            # The worker is still waiting for this barrier's acknowledgement.
            # Kill before acknowledging, so another model step cannot start.
            tree.kill()
            recorder.record(
                recorder="supervisor",
                type="supervisor.action",
                actor=supervisor,
                payload={
                    "action": "terminated",
                    "detail": "deadline complete-step checkpoint saved; runtime tree stopped",
                },
            )
            return "terminated", "deadline"
        if ended is None and time.monotonic() < hard_deadline:
            acknowledge()

    if deadline_requested:
        tree.kill()
        recorder.record(
            recorder="supervisor",
            type="supervisor.action",
            actor=supervisor,
            payload={"action": "terminated", "detail": "deadline reached; runtime ended"},
        )
        return "terminated", "deadline"
    if ended.get("reason") == "operator_pause" and stop_sent:
        recorder.record(
            recorder="supervisor",
            type="supervisor.action",
            actor=supervisor,
            payload={"action": "paused", "detail": "operator pause at complete-step barrier"},
        )
        return "paused", "operator checkpoint pause"
    if ended.get("reason") == "max_steps":
        recorder.record(
            recorder="supervisor",
            type="supervisor.action",
            actor=supervisor,
            payload={"action": "limit_enforced", "detail": "max_steps limit reached"},
        )
        return "terminated", "max_steps"
    if ended.get("reason") == "model_error":
        return "failed", "the model server could not produce a reply"
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
