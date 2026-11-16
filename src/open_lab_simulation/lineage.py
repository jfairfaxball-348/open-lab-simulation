"""Operator-only data inheritance from verified final protected snapshots."""
from __future__ import annotations

import json
import re
import shutil
from contextlib import contextmanager
from pathlib import Path

from .checkpoints import FILES, latest_checkpoint, restore_workspace, run_lock
from .contracts import Contracts, canonical_json, jsonl_lines, sha256_text
from .paths import is_link
from .world import file_sha256

SLUG = re.compile(r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$")
VERSION = "workspace-memory-v1"


@contextmanager
def source(archive, root):
    from .archive import verify_archive

    archive = Path(archive).absolute()
    if (archive.parent.parent != root / "runs" or not SLUG.fullmatch(archive.name)
            or not SLUG.fullmatch(archive.parent.name) or not archive.is_dir()
            or any(is_link(p) for p in (root / "runs", archive.parent, archive))):
        raise ValueError("lineage source must be a run archive under this data root without links")
    with run_lock(archive):
        problems = verify_archive(archive)
        if problems:
            raise ValueError(f"lineage source failed verification: {problems[0]}")
        summary = json.loads((archive / "summary.json").read_text(encoding="utf-8"))
        if summary["final_state"] != "completed":
            raise ValueError("lineage source must be completed")
        events = [json.loads(line) for line in jsonl_lines(
            (archive / "events.jsonl").read_text(encoding="utf-8"))]
        metadata, state, rejected = latest_checkpoint(
            archive, events, Contracts(archive / "contracts"))
        final_work = max((e["seq"] for e in events if e["type"] in (
            "model.call", "agent.output", "tool.result")), default=0)
        if (rejected or metadata["reason"] != "pre_shutdown"
                or metadata["last_event_seq"] < final_work):
            raise ValueError("lineage requires the verified final complete-step checkpoint")
        yield Source(archive, metadata, state)


class Source:
    def __init__(self, archive, metadata, state):
        self.archive, self.metadata, self.state = archive, metadata, state

    def install(self, child, workspace):
        ident = self.metadata["checkpoint_id"]
        origin = self.archive / "checkpoints" / ident
        evidence = child / "lineage-source" / "checkpoints" / ident
        evidence.mkdir(parents=True)
        for filename in ("checkpoint.json", FILES["workspace"], FILES["memory"]):
            shutil.copyfile(origin / filename, evidence / filename)
        document = {
            "version": VERSION, "parent_experiment_id": self.archive.parent.name,
            "parent_run_id": self.archive.name, "child_experiment_id": child.parent.name,
            "child_run_id": child.name, "checkpoint_id": ident,
            "parent_events_sha256": file_sha256(self.archive / "events.jsonl"),
            "checkpoint_metadata_sha256": file_sha256(evidence / "checkpoint.json"),
            "memory_present": self.state["memory_present"], "inherits": ["workspace", "memory"],
            "workspace_sha256": file_sha256(evidence / FILES["workspace"]),
            "memory_sha256": file_sha256(evidence / FILES["memory"]),
        }
        data = canonical_json(document).encode("utf-8")
        (child / "lineage.json").write_bytes(data)
        restore_workspace(child / "lineage-source", self.metadata,
                          {"memory_present": self.state["memory_present"]}, workspace)
        children = self.archive / "children"
        if is_link(children):
            raise ValueError("lineage child references cannot use links")
        children.mkdir(exist_ok=True)
        target = children / f"{child.name}.json"
        if target.exists():
            raise ValueError("lineage child reference already exists")
        temporary = children / f".{child.name}.tmp"
        backlink = {"version": VERSION, "child_experiment_id": child.parent.name,
                    "child_run_id": child.name, "lineage_sha256": sha256_text(data.decode())}
        with temporary.open("xb") as handle:
            handle.write(canonical_json(backlink).encode("utf-8"))
        temporary.rename(target)
        return sha256_text(data.decode())


def verify(archive, events):
    """Verify portable copied evidence; relatives need not be present."""
    children = archive / "children"
    try:
        if is_link(children):
            raise ValueError("linked child references")
        if children.exists():
            for child in children.iterdir():
                if is_link(child) or not child.is_file():
                    raise ValueError("invalid child reference file")
                reference = json.loads(child.read_text(encoding="utf-8"))
                if (set(reference) != {"version", "child_experiment_id", "child_run_id",
                                       "lineage_sha256"} or reference["version"] != VERSION
                        or child.name != reference["child_run_id"] + ".json"
                        or not SLUG.fullmatch(reference["child_run_id"])
                        or not SLUG.fullmatch(reference["child_experiment_id"])
                        or not re.fullmatch(r"[0-9a-f]{64}", reference["lineage_sha256"])):
                    raise ValueError("invalid child reference")
    except (OSError, ValueError, KeyError, TypeError):
        return ["lineage child references are invalid"]
    path = archive / "lineage.json"
    pins = [e["payload"].get("extensions", {}).get("lineage_sha256") for e in events
            if e["type"] == "run.state_changed" and e["payload"].get("to") == "created"]
    if not path.exists() and not any(pins):
        return []
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
        if (value.get("version") != VERSION or value.get("inherits") != ["workspace", "memory"]
                or type(value.get("memory_present")) is not bool
                or value.get("child_run_id") != archive.name
                or value.get("child_experiment_id") != events[0]["experiment_id"]
                or pins != [sha256_text(raw.decode("utf-8"))]
                or not all(isinstance(value.get(k), str) and SLUG.fullmatch(value[k]) for k in (
                    "parent_experiment_id", "parent_run_id", "checkpoint_id"))
                or not re.fullmatch(r"ckpt-[0-9]{8,}", value["checkpoint_id"])):
            raise ValueError("lineage identity or launch pin differs")
        folder = archive / "lineage-source/checkpoints" / value["checkpoint_id"]
        metadata = json.loads((folder / "checkpoint.json").read_text(encoding="utf-8"))
        if (file_sha256(folder / "checkpoint.json") != value["checkpoint_metadata_sha256"]
                or metadata["run_id"] != value["parent_run_id"]
                or metadata["experiment_id"] != value["parent_experiment_id"]
                or metadata["checkpoint_id"] != value["checkpoint_id"]):
            raise ValueError("lineage source metadata differs")
        components = {c["name"]: c for c in metadata["components"]}
        for name in ("workspace", "memory"):
            component = components[name]
            target = folder / FILES[name]
            if (component["path"] != FILES[name] or target.stat().st_size != component["size_bytes"]
                    or file_sha256(target) != component["sha256"]
                    or file_sha256(target) != value[f"{name}_sha256"]):
                raise ValueError("lineage source bytes differ")
    except (OSError, ValueError, KeyError, TypeError):
        return ["lineage evidence is missing, invalid or differs from its launch pin"]
    return []
