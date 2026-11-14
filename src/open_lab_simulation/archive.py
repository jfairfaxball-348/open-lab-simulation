"""Check that a run archive is complete and internally consistent.

Uses only what is inside the archive: its own contracts, event log, artifact
store and summary. It needs neither the workspace nor the world release.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import policies
from .artifacts import read_index
from .checkpoints import CheckpointError, checkpoint_events, pruned_ids, read_checkpoint
from .contracts import Contracts, canonical_json, sha256_text, validate_event_stream
from .world import file_sha256


def verify_archive(
    archive: Path, *, checkpoints: bool = True, allow_interrupted: bool = False,
) -> list[str]:
    """Problems found in a run archive; an empty list means it is intact."""
    try:
        contracts = Contracts(archive / "contracts")
        raw = (archive / "events.jsonl").read_bytes()
        if not raw.endswith(b"\n"):
            return ["the event log has a partial final line"]
        lines = raw.decode("utf-8").splitlines()
        config = json.loads((archive / "experiment.json").read_text(encoding="utf-8"))
        world = json.loads((archive / "world-release.json").read_text(encoding="utf-8"))
        summary_path = archive / "summary.json"
        summary = (
            json.loads(summary_path.read_text(encoding="utf-8"))
            if summary_path.exists() or not allow_interrupted else None
        )
    except (OSError, ValueError, KeyError) as error:
        return [f"archive is incomplete or unreadable ({type(error).__name__})"]

    problems = validate_event_stream(contracts, lines)
    problems += [f"experiment.json: {p}" for p in contracts.validate(config, "experiment-config")]
    problems += [f"world-release.json: {p}" for p in contracts.validate(world, "world-release")]
    if problems:
        return problems
    if not lines:
        return [*problems, "the event log is empty"]

    events = [json.loads(line) for line in lines]
    policy = None
    policy_pins = [e["payload"].get("extensions", {}).get("policy_sha256") for e in events
                   if e["type"] == "run.state_changed" and e["payload"].get("to") == "starting"]
    if any(policy_pins):
        try:
            policy = json.loads((archive / "policy.json").read_text(encoding="utf-8"))
            model = json.loads((archive / "model.json").read_text(encoding="utf-8"))
            if policy_pins != [policies.digest(policy)]:
                problems.append("runtime policy differs from its launch digest")
            if policy.get("experiment_sha256") != policies.digest(config):
                problems.append("experiment differs from frozen runtime policy")
            if model.get("runtime_policy") != policy:
                problems.append("model and runtime policy differ")
        except (OSError, ValueError, TypeError):
            problems.append("runtime policy is missing or unreadable")
    for event in events:
        ext = event["payload"].get("extensions", {})
        if event["type"] == "recovery" and "runtime_policy" in ext:
            restored = ext["runtime_policy"]
            if (ext.get("policy_sha256") != policies.digest(restored)
                    or restored.get("experiment_sha256") != policies.digest(config)
                    or (policy is not None and policy != restored)):
                problems.append("recovery runtime policy differs")
            if policy is None:
                policy = restored
        if ext.get("stage") == "model_input" and policy is not None:
            if ext.get("policy_sha256") != policies.digest(policy):
                problems.append("model input policy differs")
            context = ext.get("context", {})
            if "after_sha256" in context and context["after_sha256"] != policies.digest(
                ext.get("request", {}).get("messages")
            ):
                problems.append("model input context digest differs")
    pins = [e["payload"].get("extensions", {}).get("model_sha256") for e in events
            if e["type"] == "run.state_changed"
            and e["payload"].get("extensions", {}).get("stage") == "model_provenance"]
    if pins:
        try:
            model = json.loads((archive / "model.json").read_text(encoding="utf-8"))
            if pins != [sha256_text(canonical_json(model))]:
                problems.append("model provenance differs from its recorded digest")
            selected = config.get("extensions", {}).get("model_catalog")
            if model.get("catalog") != selected:
                problems.append("model catalog differs from experiment")
            if selected:
                raw_catalog = (archive / "model-catalog.json").read_bytes()
                expected = canonical_json(selected["document"])
                if (raw_catalog != expected.encode("utf-8")
                        or sha256_text(expected) != selected["sha256"]):
                    problems.append("model catalog differs from its pin")
        except (OSError, ValueError, KeyError, TypeError):
            problems.append("model provenance is missing or unreadable")
    last = next((e for e in reversed(events) if e["type"] == "run.state_changed"), events[-1])
    if summary is not None and (
        last["type"] != "run.state_changed" or last["payload"].get("to") != summary["final_state"]
    ):
        problems.append("the log does not end in the final state the summary gives")
    if summary is not None and summary["event_count"] != len(events):
        problems.append("the summary's event count differs from the log")
    if (events[0]["experiment_id"], events[0]["run_id"]) != (
        config["experiment_id"], summary["run_id"] if summary else archive.name
    ):
        problems.append("the log belongs to a different run")

    for metadata in read_index(archive):
        label = f"artifact {metadata.get('artifact_id')}"
        issues = contracts.validate(metadata, "artifact-metadata")
        if issues:
            problems.append(f"{label}: {issues[0]}")
            continue
        seq = metadata["observed_at_seq"]
        event = events[seq] if seq < len(events) else None
        if not event or event["payload"].get("artifact_id") != metadata["artifact_id"]:
            problems.append(f"{label}: no matching event at seq {seq}")
        if metadata.get("preserved", True):
            stored = archive / metadata["archive_path"]
            if not stored.is_file():
                problems.append(f"{label}: preserved content is missing")
            elif file_sha256(stored) != metadata["sha256"]:
                problems.append(f"{label}: preserved content does not match its digest")
    if checkpoints:
        pruned = pruned_ids(events)
        for event in checkpoint_events(events):
            ident = event["payload"]["checkpoint_id"]
            if ident in pruned:
                continue
            try:
                metadata, runtime = read_checkpoint(archive, event, events, contracts)
                if policy and runtime.get("policy_sha256"):
                    if runtime["policy_sha256"] != policies.digest(policy):
                        problems.append(f"{ident}: checkpoint policy differs")
                    try:
                        policies.validate_state(runtime.get("policy_state"), policy)
                    except ValueError as error:
                        problems.append(f"{ident}: {error}")
                if metadata["extensions"]["world_digest"] != world["content_digest"]["value"]:
                    problems.append(f"{ident}: checkpoint world differs")
            except CheckpointError as error:
                problems.append(str(error))
    return problems
