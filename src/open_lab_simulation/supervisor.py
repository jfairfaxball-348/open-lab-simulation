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

from . import (
    __version__,
    chatgpt,  # noqa: F401 - registers the ChatGPT plan-usage adapter
    container,
    faults,
    gateway,
    gemini,  # noqa: F401 - registers the Gemini adapter
    ollama,
    policies,
    providers,
    recovery,
    resources,
)
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
from .contracts import jsonl_lines
from .model import SCRIPTED_MODEL_ID
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
    if config.get("capabilities", {}).get("agent_created_tools"):
        return "agent-created tools are not implemented"
    tier = world.manifest["environment"]["isolation"]["minimum_tier"]
    if config["permissions"].get("subprocess") and tier != "container":
        return "running agent-supplied code needs the container isolation tier"
    if config["permissions"]["network"] == "allowlist":
        return "network access cannot be granted: this runtime cannot enforce an allowlist"
    if tier == "container":
        for ident, (_, manifest) in sorted(world.capabilities.items()):
            needed = manifest["permissions"]
            if needed["network"] != "none" or needed.get("gpu"):
                return (f"capability '{ident}' needs network or GPU access, which the "
                        "container tier cannot provide")
    return None


def _isolation(config: dict[str, Any], world: World) -> dict[str, Any]:
    """Describe the isolation every capability call of this run will get, or refuse."""
    if world.manifest["environment"]["isolation"]["minimum_tier"] != "container":
        return {"tier": "process"}
    try:
        limits = container.container_limits(config["limits"].get("resources"))
        described = container.Backend().probe()
    except container.ContainerUnavailable as error:
        raise LaunchError(f"the container isolation tier is not available: {error}") from None
    return {**described, "distro": container.DISTRO, "limits": limits}


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


def _resolve_model(config: dict[str, Any], world: World, root: Path) -> dict[str, Any]:
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

    try:
        providers, selected_catalog = gateway.catalog(config, world)
    except ValueError as error:
        raise LaunchError(str(error)) from None
    wanted_provider = orchestrator.get("provider_id")
    found = [
        (provider_id, document["provider"], model)
        for provider_id, document in providers.items()
        if wanted_provider in (None, provider_id)
        for model in document["models"]
        if model["id"] == model_id
    ]
    if len(found) != 1:
        raise LaunchError(f"the world does not define exactly one model '{model_id}'")
    provider_id, provider, model = found[0]
    api = provider["api"]
    if "orchestrator" not in model["roles"]:
        raise LaunchError(f"model '{model_id}' is not an orchestrator")
    # Authority comes from the experiment: it must grant the model and the loopback call.
    permissions = config["permissions"]
    if permissions["network"] != "loopback" or model_id not in permissions.get("models", []):
        raise LaunchError(
            f"the experiment must grant network 'loopback' and list '{model_id}' in "
            "permissions.models"
        )
    if provider["locality"] == "external":
        return _resolve_hosted(config, root, provider_id, provider, model, selected_catalog)
    try:
        settings = recovery.policy(config)
        identity = gateway.identify(
            provider, model["provider_ref"],
            timeout=settings["health_check_timeout_s"] if settings else 30,
        )
    except (ollama.OllamaError, ValueError) as error:
        raise LaunchError(str(error)) from None
    if settings and not identity.get("digest"):
        raise LaunchError("model recovery requires an attestable model digest")
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
    try:
        resolved.update(gateway.provenance(
            provider, model, resolved["sampling"], identity, selected_catalog,
        ))
    except ValueError as error:
        raise LaunchError(str(error)) from None
    if "context_window_tokens" in model:
        resolved["context_tokens"] = model["context_window_tokens"]
    return resolved


def _resolve_hosted(config, root, provider_id, provider, model, selected_catalog):
    """A hosted model, reached only through the supervisor's provider broker (ADR-0016)."""
    if recovery.policy(config):
        raise LaunchError("automatic recovery is not supported for hosted providers")
    extensions = config.get("extensions", {})
    campaign = extensions.get("benchmark", {}).get("campaign_id")
    inference = extensions.get("hosted_inference", {})
    sampling = config["orchestrator"].get("sampling", {})
    try:
        checked = providers.launch_check(root, provider_id, campaign=campaign)
        adapter = checked["adapter"]
        support = adapter.settings_support(sampling, inference)
    except (providers.ProviderError, ValueError) as error:
        raise LaunchError(str(error)) from None
    if model["status"] != "available":
        raise LaunchError("the model must be approved as available")
    profile = checked["profile"]
    resolved = {
        "kind": "openai_chat",
        "model_id": model["id"],
        "provider_id": provider_id,
        "real_model": True,
        "base_url": None,
        "provider_ref": model["provider_ref"],
        "sampling": sampling,
        "identity": {"provider_ref": model["provider_ref"], "digest": None,
                     "identity_limit": "hosted model: no weight digest; per-call model "
                                       "versions are in provider-ledger.jsonl"},
        "provider_kind": provider["kind"],
        "catalog": selected_catalog,
        "model_manifest": model,
        "provider_manifest": provider,
        "hosted": {
            "adapter": {"id": adapter.id, "version": adapter.version, "host": adapter.host},
            "inference": inference,
            "campaign": campaign,
            "limits": profile["limits"],
            "max_attempts_per_step": profile["max_attempts_per_step"],
            "min_interval_s": profile["min_interval_s"],
            "request_timeout_s": profile["request_timeout_s"],
        },
        "settings": {
            "context_guard_tokens": model.get("context_window_tokens"),
            "effective_context_tokens": None,
            "context_limit": "client guard only; provider context is not set or attested",
            "sampling_requested": sampling,
            "sampling_effective": support,
            "output_tokens_requested": sampling.get("max_output_tokens"),
            "offload": None,
            "repair_policy": "syntactic-tool-repair-v1",
            "request_attempts_max": profile["max_attempts_per_step"],
        },
    }
    if "context_window_tokens" in model:
        resolved["context_tokens"] = model["context_window_tokens"]
    return resolved


def _prepare(
    config_path: Path, root: Path
) -> tuple[bytes, dict[str, Any], World, dict[str, Any], dict[str, Any]]:
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
    try:
        recovery.policy(config)
        resources.validate(config)
        faults.enabled(config)
    except ValueError as error:
        raise LaunchError(str(error)) from None
    reason = _unsupported(config, world)
    if reason:
        raise LaunchError(f"not supported by this runtime: {reason}")
    try:
        Registry(world.directory, world.capabilities)
    except RegistryError as error:
        raise LaunchError(str(error)) from None
    isolation = _isolation(config, world)
    model = _resolve_model(config, world, root)
    try:
        model["runtime_policy"] = policies.resolve(
            config, continuation_message(config, world), model.get("context_tokens"),
        )
    except ValueError as error:
        raise LaunchError(str(error)) from None
    if "settings" in model:
        model["settings"]["context_guard_tokens"] = model["runtime_policy"]["context_guard_tokens"]
    return raw, config, world, model, isolation


def _new_run_id() -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dt%H%M%Sz")
    return f"run-{stamp}-{secrets.token_hex(3)}"


def run_experiment(
    config_path: Path, root: Path | None = None, *, pause_after_steps: int | None = None
) -> RunResult:
    """Launch one fresh run. Operator pauses require configured checkpointing."""
    root = (root or data_root()).resolve()
    raw, config, world, model, isolation = _prepare(config_path, root)
    if pause_after_steps is not None and (pause_after_steps < 1 or "checkpointing" not in config):
        raise LaunchError("pausing needs checkpointing and a positive pause-after-steps value")
    run_id = _new_run_id()
    archive = root / "runs" / config["experiment_id"] / run_id
    workspace = root / "workspaces" / run_id
    archive.mkdir(parents=True)
    workspace.mkdir(parents=True)
    with run_lock(archive):
        result = _run_prepared(
            raw, config, world, model, archive, workspace, isolation=isolation,
            pause_after_steps=pause_after_steps,
        )
        return _continue_recovery(result, root, config, pause_after_steps)


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
            result = _resume_locked(archive, root, pause_after_steps=pause_after_steps)
            config = json.loads((archive / "experiment.json").read_text(encoding="utf-8"))
            return _continue_recovery(result, root, config, pause_after_steps)
    except (CheckpointError, OSError, ValueError, KeyError) as error:
        raise LaunchError(f"resume refused: {error}") from None


def _resume_locked(archive, root, *, pause_after_steps=None, source="operator", stale=None):
    recovery_started = time.monotonic()
    interrupted = not (archive / "summary.json").exists()
    problems = verify_archive(archive, checkpoints=False, allow_interrupted=source == "startup")
    if problems:
        raise LaunchError(f"run archive cannot be resumed: {problems[0]}")
    raw, config, world, model, isolation = _prepare(archive / "experiment.json", root)
    settings = recovery.policy(config)
    events = [json.loads(line) for line in
              jsonl_lines((archive / "events.jsonl").read_text(encoding="utf-8"))]
    if interrupted:
        if not settings or not settings["auto_resume"]:
            raise LaunchError("interrupted run did not opt into startup recovery")
        summary = {
            "final_state": next(e["payload"]["to"] for e in reversed(events)
                                if e["type"] == "run.state_changed"),
            "started_at": events[0]["timestamp"],
            "world": {"content_digest": json.loads(
                (archive / "world-release.json").read_text()
            )["content_digest"]["value"]},
            "elapsed_s": recovery.interrupted_elapsed(archive),
        }
        if summary["final_state"] in ("completed", "terminated", "paused"):
            raise LaunchError("interrupted archive already records a stopped run; review required")
    else:
        summary = json.loads((archive / "summary.json").read_text(encoding="utf-8"))
        if summary["final_state"] not in ("paused", "terminated", "failed"):
            raise LaunchError("only a stopped, incomplete run can be resumed")
        if source == "operator" and summary["final_state"] == "failed" and settings:
            source = "operator_retry"
    if source != "operator" and (
        not settings or recovery.restarts_used(events) >= settings["max_restarts"]
    ):
        raise LaunchError("recovery retry budget exhausted")
    if "checkpointing" not in config:
        raise LaunchError("run did not enable checkpointing")
    if world.digest != summary["world"]["content_digest"]:
        raise LaunchError("resuming requires the original world digest")
    prior_model = json.loads((archive / "model.json").read_text(encoding="utf-8"))
    if model["real_model"] and not model.get("identity", {}).get("digest"):
        raise LaunchError("resuming requires an attestable model digest")
    comparable = dict(model)
    if "runtime_policy" not in prior_model:
        comparable.pop("runtime_policy", None)
    if prior_model.get("adapter") and comparable != prior_model:
        raise LaunchError("resuming requires the original model digest, catalog and settings")
    if model.get("identity", {}).get("digest") != prior_model.get("identity", {}).get(
        "digest"
    ):
        raise LaunchError("resuming requires the original model digest")
    metadata, state, rejected = latest_checkpoint(archive, events, world.contracts)
    if state.get("policy_sha256"):
        if state["policy_sha256"] != policies.digest(model["runtime_policy"]):
            raise LaunchError("checkpoint runtime policy differs")
        policies.validate_state(state.get("policy_state"), model["runtime_policy"])
    elif model["runtime_policy"]["id"] != policies.LEGACY:
        raise LaunchError("quiet policy needs checkpoint scheduling state")
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
    consumed = recovery.steps_used(events)
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
    elapsed += time.monotonic() - recovery_started
    if elapsed >= config["limits"]["wall_clock_s"]:
        raise LaunchError("run has exhausted its original time budget during recovery")
    resume = {
        "metadata": metadata,
        "state": state,
        "rejected": rejected,
        "elapsed": elapsed,
        "consumed": consumed,
        "summary": summary,
        "source": source,
        "stale": stale,
    }
    return _run_prepared(
        raw,
        config,
        world,
        model,
        archive,
        workspace,
        isolation=isolation,
        resume=resume,
        pause_after_steps=pause_after_steps,
    )

def _recovery_action(archive, stage, detail):
    """Append only under the caller's writer lock; keep the summary consistent."""
    from .contracts import Contracts

    summary = json.loads((archive / "summary.json").read_text(encoding="utf-8"))
    recorder = Recorder(archive / "events.jsonl", Contracts(archive / "contracts"),
                        summary["experiment_id"], archive.name,
                        append=True, elapsed_s=summary["elapsed_s"])
    try:
        recorder.record(recorder="supervisor", type="supervisor.action",
                        actor={"kind": "supervisor"}, payload={
                            "action": "limit_enforced", "detail": detail,
                            "extensions": {"stage": stage},
                        })
        recorder.sync()
        summary.update(event_count=recorder.count, events_by_type=recorder.by_type,
                       last_event_sha256=recorder.last_event_sha256)
        (archive / "summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                              encoding="utf-8", newline="\n")
    finally:
        recorder.close()


def _continue_recovery(result, root, config, pause_after_steps):
    settings = recovery.policy(config)
    if settings is None:
        return result
    while result.state == "failed" and (
        result.reason.startswith("the agent runtime exited unexpectedly")
        or result.reason in (
            "the model server could not produce a reply", "model health check failed",
        )
    ):
        events = [json.loads(line) for line in
                  jsonl_lines((result.archive / "events.jsonl").read_text(encoding="utf-8"))]
        if recovery.restarts_used(events) >= settings["max_restarts"]:
            _recovery_action(result.archive, "recovery_exhausted",
                             "recovery retry budget exhausted")
            break
        try:
            result = _resume_locked(result.archive, root, pause_after_steps=pause_after_steps,
                                    source="runtime_crash")
        except (LaunchError, OSError, ValueError, KeyError) as error:
            _recovery_action(result.archive, "recovery_refused", str(error))
            break
    # A refusal/exhaustion action changes the count after the stopped segment.
    result.events = json.loads((result.archive / "summary.json").read_text())["event_count"]
    return result


def recover_interrupted(root: Path | None = None):
    """Startup entrypoint: recover only explicitly opted-in interrupted archives."""
    root = (root or data_root()).resolve()
    results, refused = [], []
    for archive in sorted((root / "runs").glob("*/*")):
        if not archive.is_dir() or (archive / "summary.json").exists():
            continue
        try:
            if (
                archive.is_symlink() or archive.parent.is_symlink()
                or archive.is_junction() or archive.parent.is_junction()
                or not _SLUG.fullmatch(archive.name) or not _SLUG.fullmatch(archive.parent.name)
            ):
                raise LaunchError("invalid archive path")
            config = json.loads((archive / "experiment.json").read_text(encoding="utf-8"))
            settings = recovery.policy(config)
            if not settings or not settings["auto_resume"]:
                continue
            with run_lock(archive, recover_stale=True) as stale:
                try:
                    result = _resume_locked(archive, root, source="startup", stale=stale)
                except (LaunchError, OSError, ValueError, KeyError) as error:
                    _interrupted_refusal(archive, str(error))
                    raise
                results.append(_continue_recovery(result, root, config, None))
        except (LaunchError, OSError, ValueError, KeyError) as error:
            refused.append((archive.name, str(error)))
    return results, refused


def _interrupted_refusal(archive, detail):
    # A live/unverifiable writer or corrupt log cannot safely accept an event.
    # This path runs only after exclusive ownership and complete-log validation.
    if verify_archive(archive, checkpoints=False, allow_interrupted=True):
        return
    from .contracts import Contracts

    events = [json.loads(line) for line in
              jsonl_lines((archive / "events.jsonl").read_text(encoding="utf-8"))]
    recorder = Recorder(archive / "events.jsonl", Contracts(archive / "contracts"),
                        events[0]["experiment_id"], archive.name, append=True,
                        elapsed_s=events[-1].get("monotonic_ms", 0) / 1000)
    try:
        recorder.record(recorder="supervisor", type="supervisor.action",
                        actor={"kind": "supervisor"}, payload={
                            "action": "limit_enforced", "detail": detail,
                            "extensions": {"stage": "recovery_refused"},
                        })
        recorder.sync()
    finally:
        recorder.close()


def _run_prepared(
    raw, config, world, model, archive, workspace, *, isolation, resume=None,
    pause_after_steps=None,
):
    run_id, experiment_id = archive.name, config["experiment_id"]
    contained = isolation["tier"] == "container"
    if resume is None:
        (archive / "experiment.json").write_bytes(raw)
        shutil.copyfile(world.directory / "world-release.json", archive / "world-release.json")
        shutil.copytree(world.directory / "contracts", archive / "contracts")
        if model.get("catalog"):
            (archive / "model-catalog.json").write_text(
                gateway.canonical_json(model["catalog"]["document"]),
                encoding="utf-8", newline="\n",
            )
        (archive / "policy.json").write_text(
            gateway.canonical_json(model["runtime_policy"]), encoding="utf-8", newline="\n",
        )
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
            payload={"from": state, "to": to, "reason": reason,
                     **({"extensions": {
                         "policy_sha256": policies.digest(model["runtime_policy"]),
                         **({"stage": "model_provenance", "model_sha256": gateway.sha256_text(
                             gateway.canonical_json(model))} if model.get("adapter") else {}),
                     }} if to == "starting" else {})},
        )
        state = to

    # Resumed segments may run on a different backend build; each one is kept.
    described_isolation = json.dumps(isolation, indent=2) + "\n"
    if not resume:
        (archive / "isolation.json").write_text(
            described_isolation, encoding="utf-8", newline="\n"
        )
    if resume:
        segments = archive / "segments"
        segments.mkdir(exist_ok=True)
        (segments / f"isolation-{recorder.count:08d}.json").write_text(
            described_isolation, encoding="utf-8", newline="\n"
        )
        if (archive / "summary.json").exists():
            shutil.copyfile(archive / "summary.json", segments / f"{recorder.count:08d}.json")
            (archive / "summary.json").unlink()
        source = resume.get("source", "operator")
        if (archive / ".supervisor.json").exists():
            shutil.copyfile(archive / ".supervisor.json",
                            segments / f"accounting-{recorder.count:08d}.json")
        change("recovering", f"{source} checkpoint resume")
        recorder.record(recorder="supervisor", type="supervisor.action", actor=supervisor,
                        payload={"action": "resumed", "detail": f"{source} checkpoint resume",
                                 "extensions": {"stage": "recovery_restart" if source != "operator"
                                                else "manual_resume",
                                                "stale_lock_evidence": resume.get("stale")}})
        recorder.sync()
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
                    "runtime_policy": model["runtime_policy"],
                    "policy_sha256": policies.digest(model["runtime_policy"]),
                },
            },
        )
        scanner.scan({"kind": "system"})
    else:
        initial_entries = sorted(p.relative_to(workspace).as_posix()
                                 for p in workspace.rglob("*"))
        recorder.record(
            recorder="experiment_manager",
            type="run.state_changed",
            actor=system,
            payload={"to": "created", "reason": "launch", "extensions": {
                "initial_workspace": {"files": initial_entries,
                                      "sha256": policies.digest(initial_entries),
                                      "memory_present": (workspace / ".memory.sqlite").exists()},
            }},
        )
        change("starting", "launch", by="experiment_manager")

    notes: list[str] = []
    tree: ProcessTree | None = None
    stderr_log = (archive / "runtime.stderr.log").open("ab" if resume else "wb")
    final, reason = "failed", "supervisor error"
    accounting = None
    controller = None
    broker = None
    try:
        worker_model = model
        if model.get("hosted"):
            # Archives always live at <root>/runs/<experiment>/<run>.
            broker = providers.Broker(archive.parent.parent.parent, providers.Plan(
                provider_id=model["provider_id"], model_ref=model["provider_ref"],
                sampling=model["sampling"], inference=model["hosted"]["inference"],
                profile={"limits": model["hosted"]["limits"],
                         **{k: model["hosted"][k] for k in providers.PROFILE_DEFAULTS}},
                run_id=run_id, campaign=model["hosted"]["campaign"], archive=archive,
            )).start()
            # The run token reaches only the runtime's stdin, never model.json or events.
            worker_model = {**model, "base_url": broker.url, "auth_token": broker.token}
        controller = resources.Controller(config, workspace, world, recorder)
        if recovery.policy(config):
            accounting = recovery.Accounting(archive, elapsed_offset)
        tree = ProcessTree(
            WORKER_COMMAND,
            stdin=-1,
            stdout=-1,
            stderr=stderr_log,
            env=minimal_environment(),
            resources=config["limits"].get("resources"),
        )
        controller.installed()
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
            "benchmark_fault": faults.enabled(config),
            "permissions": config["permissions"],
            "allow": capabilities.get("allow"),
            "deny": capabilities.get("deny"),
            "model": worker_model,
            "actor": agent,
            "isolation": (
                {"tier": "container", "run_id": run_id, "distro": isolation["distro"],
                 "image": isolation["image"], "limits": isolation["limits"]}
                if contained else {"tier": "process"}
            ),
        }
        if resume:
            spec["runtime_state"] = resume["state"]
        process = tree.process
        if accounting:
            accounting.attach(process.pid)
            spec["reserve_steps"] = True
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
            accounting=accounting,
            health=(recovery.HealthMonitor(model, recovery.policy(config))
                    if accounting and model["real_model"] else None),
            controller=controller, runtime_policy=model["runtime_policy"],
            restored_policy=resume is not None,
            benchmark_fault=faults.enabled(config),
            fault_used=bool(resume and any(
                json.loads(line)["payload"].get("extensions", {}).get("stage") == "benchmark_fault"
                for line in jsonl_lines((archive / "events.jsonl").read_text(encoding="utf-8")))),
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
        if controller is not None:
            controller.action("terminated", "monitoring", "resource setup or monitoring failed")
    finally:
        if tree is not None:
            tree.kill()
        if broker is not None:
            broker.stop()
            if broker.exhausted:
                recorder.record(
                    recorder="supervisor", type="supervisor.action", actor=supervisor,
                    payload={"action": "limit_enforced",
                             "detail": f"approved provider budget '{broker.exhausted}' exhausted; "
                                       "no further provider request was sent",
                             "extensions": {"stage": "provider_budget",
                                            "limit": f"provider_{broker.exhausted}"}},
                )
        if contained:
            # Containers outlive the runtime's process tree; nothing of the run may.
            try:
                removed = container.Backend(isolation["distro"], isolation["image"]).sweep(run_id)
            except container.ContainerUnavailable as error:
                notes.append(f"containers of the run may still exist: {error}")
                removed = []
            if removed:
                recorder.record(
                    recorder="supervisor", type="supervisor.action", actor=supervisor,
                    payload={"action": "terminated",
                             "detail": f"removed {len(removed)} container(s) still present "
                                       "after the runtime stopped",
                             "extensions": {"stage": "container_cleanup"}},
                )
        if tree is not None:
            if controller is not None:
                try:
                    controller.tick(tree, {}, force=True)
                except (OSError, ValueError):
                    controller.action("terminated", "monitoring",
                                      "final resource sample unavailable")
            tree.close()
        stderr_log.close()
        if accounting:
            accounting.close()

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
        "isolation": {key: isolation[key] for key in ("tier", "image") if key in isolation},
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
    accounting=None,
    health=None,
    controller=None,
    runtime_policy=None,
    restored_policy=False,
    benchmark_fault=False,
    fault_used=False,
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
    denied_calls: set[str] = set()
    ended: dict[str, Any] | None = None
    last_seq: int | None = None
    next_checkpoint = time.monotonic()
    pause_requested = False
    idle_recorded = False
    stop_sent = False
    last_denial = None
    inject_fault = False

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
        nonlocal ended, last_seq, next_checkpoint, stop_sent, deadline_checkpointed, last_denial
        nonlocal idle_recorded, fault_used, inject_fault
        inject_fault = False
        last_seq = None
        last_denial = None
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
                (checkpointing is None and (runtime_policy is None
                 or runtime_policy["id"] != policies.QUIET))
                or open_calls
                or not isinstance(runtime.get("messages"), list)
                or any(not isinstance(runtime.get(k), int) or runtime[k] < 0 for k in fields)
            ):
                return refuse("invalid checkpoint state barrier")
            if runtime_policy:
                try:
                    if runtime.get("policy_sha256") != policies.digest(runtime_policy):
                        raise ValueError("runtime policy digest differs")
                    policies.validate_state(runtime.get("policy_state"), runtime_policy)
                except ValueError as error:
                    return refuse(str(error))
                if runtime["policy_state"]["idle"] and not idle_recorded:
                    recorder.record(
                        recorder="supervisor", type="supervisor.action", actor=supervisor,
                        payload={"action": "limit_enforced", "detail": "quiet reply budget reached",
                                 "extensions": {"stage": "runtime_idle", "wake": "none",
                                                "restored": restored_policy,
                                                "policy_state": runtime["policy_state"]}},
                    )
                    idle_recorded = True
            stop_sent = deadline_requested or pause_requested or (
                pause_after_steps is not None and runtime["steps"] >= pause_after_steps
            )
            now = time.monotonic()
            if checkpointing is not None and (
                stop_sent or message.get("final") or now >= next_checkpoint
            ):
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
        if isinstance(message, dict) and isinstance(message.get("benchmark_fault"), dict):
            data = message["benchmark_fault"]
            if (stopped or not benchmark_fault or len(open_calls) != 1
                    or data.get("capability_id") not in open_calls.values()
                    or denied_calls):
                return refuse("invalid benchmark execution admission")
            if not fault_used:
                inject_fault = True
                fault_used = True
                recorder.record(
                    recorder="supervisor", type="supervisor.action", actor=supervisor,
                    payload={"action": "limit_enforced", "detail": "registered transient fault",
                             "extensions": {"stage": "benchmark_fault", **data,
                                            "correlation_id": next(iter(open_calls)),
                                            "delivered": faults.OUTCOME}},
                )
                recorder.sync()
            return None
        if isinstance(message, dict) and isinstance(message.get("model_input"), dict):
            if stopped:
                return None
            payload = message["model_input"]
            body = payload.get("request")
            if (open_calls or not isinstance(body, dict)
                    or not isinstance(body.get("messages"), list)
                    or not isinstance(payload.get("context"), dict)):
                return refuse("invalid model input request")
            event = recorder.record(
                recorder="supervisor", type="supervisor.action", actor=supervisor,
                payload={"action": "limit_enforced",
                         "detail": "context policy prepared model input",
                         "extensions": {"stage": "model_input", **payload,
                                        "policy_sha256": policies.digest(runtime_policy)}},
            )
            recorder.sync()
            last_seq = event["seq"]
            return None
        if isinstance(message, dict) and isinstance(message.get("end"), dict):
            ended = message["end"]
            return None
        if message == {"step_start": True} and accounting is not None and not stopped:
            recorder.record(recorder="supervisor", type="supervisor.action", actor=supervisor,
                            payload={"action": "limit_enforced",
                                     "detail": "reserved one model attempt against original budget",
                                     "extensions": {"stage": "model_step_reserved"}})
            recorder.sync()
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
            if controller is not None and not stopped:
                last_denial = controller.admit(target)
                if last_denial:
                    denied_calls.add(event["correlation_id"])
        elif kind == "tool.result":
            # The runtime is waiting for the acknowledgement, so whatever changed
            # in the workspace since the last scan happened during this call.
            capability_id = open_calls.pop(event["correlation_id"], None)
            denied_calls.discard(event["correlation_id"])
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
                (json.dumps({"ack": True, "seq": last_seq, "stop": stop_sent,
                             "deny": last_denial, "inject_fault": inject_fault}) + "\n").encode()
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

    def resource_stop():
        if controller is None:
            return None
        try:
            limit = controller.tick(tree, {k: v for k, v in open_calls.items()
                                           if k not in denied_calls})
        except (OSError, ValueError):
            limit = "monitoring"
        if limit is None:
            return None
        tree.kill()
        _drain(messages, lambda line: handle(line, stopped=True))
        controller.action("terminated", limit, f"resource enforcement stopped the tree: {limit}")
        return "terminated", f"resource_limit:{limit}"

    while ended is None:
        request_deadline()
        remaining = hard_deadline - time.monotonic()
        if remaining <= 0:
            return deadline_stop()
        stopped_by_resource = resource_stop()
        if stopped_by_resource:
            return stopped_by_resource
        if accounting is not None and (
            accounting.error is not None or time.monotonic() >= accounting.lease_deadline
        ):
            tree.kill()
            recorder.record(recorder="supervisor", type="supervisor.action", actor=supervisor,
                            payload={"action": "terminated",
                                     "detail": "recovery accounting failed"})
            return "failed", "recovery accounting failed"
        if health is not None:
            status = health.poll()
            if status is not None:
                recorder.record(recorder="supervisor", type="supervisor.action", actor=supervisor,
                                payload={"action": "limit_enforced",
                                         "detail": f"model health: {status}",
                                         "extensions": {"stage": "model_health", "status": status}})
                if status != "healthy":
                    tree.kill()
                    _drain(messages, lambda line: handle(line, stopped=True))
                    return "failed", "model health check failed"
        try:
            # Wake at the nominal deadline even when there is a grace allowance.
            wait = min(remaining, 0.05 if controller is not None else 0.25)
            if accounting is not None:
                wait = min(wait, max(0.001, accounting.lease_deadline - time.monotonic()))
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
            stopped_by_resource = resource_stop()
            if stopped_by_resource:
                return stopped_by_resource
            code = process.wait()
            return "failed", f"the agent runtime exited unexpectedly (status {code})"
        if time.monotonic() >= hard_deadline:
            return deadline_stop(line)
        handle(line)
        stopped_by_resource = resource_stop()
        if stopped_by_resource:
            return stopped_by_resource
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
        if ended is None and time.monotonic() < hard_deadline and (
            accounting is None or time.monotonic() < accounting.lease_deadline
        ):
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
