"""Supervisor-owned, digest-bound snapshots taken only at complete-step barriers."""

from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import tempfile
import time
import zipfile
from contextlib import closing
from pathlib import Path, PurePosixPath

from .contracts import sha256_text
from .ownership import archive_lock
from .paths import is_link
from .recorder import utc_now
from .world import file_sha256

FILES = {
    "conversation": "conversation.json",
    "memory": "memory.sqlite",
    "workspace": "workspace.zip",
    "runtime_state": "runtime_state.json",
}
MEMORY_FILES = {
    ".memory.sqlite",
    ".memory.sqlite-journal",
    ".memory.sqlite-wal",
    ".memory.sqlite-shm",
}
_CHECKPOINT = re.compile(r"ckpt-[0-9]{8,}")


class CheckpointError(ValueError):
    """A snapshot cannot be trusted or produced."""


class CheckpointTimeout(TimeoutError):
    """The supervisor's original deadline interrupted snapshot creation."""


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


run_lock = archive_lock


def _no_link(path):
    if is_link(path):
        raise CheckpointError("snapshot paths cannot be links")


def _entries(workspace):
    """Every path beneath the workspace, links included but never entered."""
    for directory, subdirectories, names in os.walk(workspace):
        for name in [*subdirectories, *names]:
            yield Path(directory, name)
        subdirectories[:] = [name for name in subdirectories
                             if not is_link(Path(directory, name))]


def _member_path(name):
    relative = PurePosixPath(name)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or "\\" in name
        or ":" in name
        or not relative.parts
    ):
        raise CheckpointError("unsafe workspace snapshot path")
    return relative


def _remove_owned_tree(path, root):
    _no_link(path)
    resolved, boundary = path.resolve(), root.resolve()
    if resolved == boundary or not resolved.is_relative_to(boundary):
        raise CheckpointError("refusing to remove a directory outside its staging root")
    shutil.rmtree(path)


def memory_backup(source: Path | None, target: Path, check=lambda: None):
    check()
    with closing(sqlite3.connect(target)) as dest:
        if source is not None:
            _no_link(source)
            with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as origin:
                if origin.execute("PRAGMA quick_check").fetchone() != ("ok",):
                    raise CheckpointError("memory integrity check failed")
                origin.backup(dest, pages=64, progress=lambda *_: check())
        else:
            dest.execute(
                "CREATE TABLE notes (key TEXT PRIMARY KEY, value TEXT NOT NULL, "
                "revision INTEGER NOT NULL)"
            )
            dest.commit()


def _workspace_snapshot(workspace, target, check):
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(_entries(workspace)):
            check()
            _no_link(path)
            name = path.relative_to(workspace).as_posix()
            _member_path(name)
            if name in MEMORY_FILES:
                continue
            if path.is_file():
                with path.open("rb") as source, bundle.open(name, "w") as dest:
                    while block := source.read(1 << 20):
                        check()
                        dest.write(block)
            elif path.is_dir():
                bundle.write(path, name)
            else:
                raise CheckpointError("workspace contains unsupported file types")


def checkpoint_events(events):
    return [e for e in events if e["type"] == "checkpoint.created"]


def pruned_ids(events):
    return {
        ident for e in checkpoint_events(events) for ident in _event_extension(e).get("pruned", [])
    }


def _event_extension(event):
    # Accept the local S006a engineering archives produced before extension
    # fields were grouped. New records use the documented payload extension.
    return event["payload"].get("extensions", event["payload"])


def write_checkpoint(
    archive,
    workspace,
    recorder,
    world,
    model_id,
    state,
    elapsed,
    reason,
    keep_last=3,
    deadline=None,
):
    def check():
        if deadline is not None and time.monotonic() >= deadline:
            raise CheckpointTimeout("deadline reached while writing checkpoint")

    parent = archive / "checkpoints"
    _no_link(parent)
    parent.mkdir(exist_ok=True)
    ident = f"ckpt-{recorder.count:08d}"
    final = parent / ident
    stage = Path(tempfile.mkdtemp(prefix=".building-", dir=parent))
    try:
        check()
        runtime = {k: state[k] for k in ("steps", "script_cursor", "call_count", "tokens_used")}
        runtime["memory_present"] = (workspace / ".memory.sqlite").is_file()
        for key in ("policy_state", "policy_sha256", "agent_tools"):
            if key in state:
                runtime[key] = state[key]
        write_json(stage / FILES["conversation"], state["messages"])
        write_json(stage / FILES["runtime_state"], runtime)
        source = workspace / ".memory.sqlite" if runtime["memory_present"] else None
        memory_backup(source, stage / FILES["memory"], check)
        _workspace_snapshot(workspace, stage / FILES["workspace"], check)
        check()
        metadata = {
            "schema": "open-lab/checkpoint-metadata",
            "schema_version": world.contracts.version("checkpoint-metadata"),
            "checkpoint_id": ident,
            "experiment_id": recorder.experiment_id,
            "run_id": recorder.run_id,
            "created_at": utc_now(),
            "reason": reason,
            "last_event_seq": recorder.count - 1,
            "progress": {
                "elapsed_s": elapsed,
                "step_count": state["steps"],
                "model_tokens_used": state["tokens_used"],
            },
            "world": {"world_id": world.manifest["world_id"], "version": world.manifest["version"]},
            "orchestrator_model_id": model_id,
            "components": [
                {
                    "name": name,
                    "path": path,
                    "sha256": file_sha256(stage / path),
                    "size_bytes": (stage / path).stat().st_size,
                }
                for name, path in FILES.items()
            ],
            "resumable": True,
            "extensions": {
                "world_digest": world.digest,
                "event_sha256": recorder.last_event_sha256,
            },
        }
        problems = world.contracts.validate(metadata, "checkpoint-metadata")
        if problems:
            raise CheckpointError(problems[0])
        write_json(stage / "checkpoint.json", metadata)
        check()
        os.replace(stage, final)
        existing = sorted(p.name for p in parent.iterdir() if _CHECKPOINT.fullmatch(p.name))
        removed = existing[:-keep_last]
        recorder.record(
            recorder="supervisor",
            type="checkpoint.created",
            actor={"kind": "supervisor"},
            payload={
                "checkpoint_id": ident,
                "extensions": {
                    "metadata_sha256": file_sha256(final / "checkpoint.json"),
                    "pruned": removed,
                },
            },
        )
        for name in removed:
            victim = parent / name
            _no_link(victim)
            if not victim.resolve().is_relative_to(parent.resolve()):
                raise CheckpointError("checkpoint pruning path escaped archive")
            _remove_owned_tree(victim, parent)
        return ident
    except CheckpointTimeout:
        if stage.exists():
            _remove_owned_tree(stage, parent)
        raise
    except (OSError, sqlite3.Error, ValueError) as error:
        if stage.exists():
            _remove_owned_tree(stage, parent)
        raise CheckpointError(f"checkpoint could not be written ({type(error).__name__})") from None


def read_checkpoint(archive, event, events, contracts):
    ident = event["payload"]["checkpoint_id"]
    if not _CHECKPOINT.fullmatch(ident):
        raise CheckpointError("invalid checkpoint identifier")
    directory = archive / "checkpoints" / ident
    try:
        _no_link(directory.parent)
        _no_link(directory)
        path = directory / "checkpoint.json"
        _no_link(path)
        if file_sha256(path) != _event_extension(event)["metadata_sha256"]:
            raise CheckpointError("checkpoint metadata digest differs")
        metadata = json.loads(path.read_text(encoding="utf-8"))
        problems = contracts.validate(metadata, "checkpoint-metadata")
        if problems or not metadata.get("resumable", True):
            raise CheckpointError("checkpoint metadata is invalid or not resumable")
        if (
            metadata["checkpoint_id"] != ident
            or metadata["run_id"] != event["run_id"]
            or metadata["experiment_id"] != event["experiment_id"]
        ):
            raise CheckpointError("checkpoint identity differs")
        seq = metadata["last_event_seq"]
        if seq >= event["seq"] or metadata["extensions"]["event_sha256"] != sha256_text(
            json.dumps(events[seq], sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        ):
            raise CheckpointError("checkpoint event boundary differs")
        components = metadata["components"]
        if len(components) != len(FILES) or {c["name"] for c in components} != FILES.keys():
            raise CheckpointError("checkpoint components are incomplete")
        for component in components:
            if component["path"] != FILES[component["name"]]:
                raise CheckpointError("invalid component path")
            path = directory / component["path"]
            _no_link(path)
            if (
                path.stat().st_size != component["size_bytes"]
                or file_sha256(path) != component["sha256"]
            ):
                raise CheckpointError("checkpoint component digest differs")
        state = json.loads((directory / FILES["runtime_state"]).read_text(encoding="utf-8"))
        state["messages"] = json.loads(
            (directory / FILES["conversation"]).read_text(encoding="utf-8")
        )
        if (
            not isinstance(state["messages"], list)
            or len(state["messages"]) < 2
            or any(
                not isinstance(state[k], int) or state[k] < 0
                for k in ("steps", "script_cursor", "call_count", "tokens_used")
            )
            or state["steps"] != metadata["progress"]["step_count"]
        ):
            raise CheckpointError("checkpoint runtime state is invalid")
        with zipfile.ZipFile(directory / FILES["workspace"]) as bundle:
            seen = set()
            for entry in bundle.infolist():
                _member_path(entry.orig_filename)
                if entry.orig_filename in seen:
                    raise CheckpointError("duplicate workspace snapshot path")
                seen.add(entry.orig_filename)
            if bundle.testzip() is not None:
                raise CheckpointError("workspace ZIP integrity check failed")
        with closing(
            sqlite3.connect((directory / FILES["memory"]).as_uri() + "?mode=ro", uri=True)
        ) as database:
            if database.execute("PRAGMA quick_check").fetchone() != ("ok",):
                raise CheckpointError("checkpoint memory is corrupt")
        return metadata, state
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        IndexError,
        sqlite3.Error,
        zipfile.BadZipFile,
    ) as error:
        raise CheckpointError(f"{ident} is not usable ({type(error).__name__})") from None


def latest_checkpoint(archive, events, contracts):
    rejected = []
    pruned = pruned_ids(events)
    for event in reversed(checkpoint_events(events)):
        ident = event["payload"]["checkpoint_id"]
        if ident in pruned:
            continue
        try:
            metadata, state = read_checkpoint(archive, event, events, contracts)
            return metadata, state, rejected
        except CheckpointError:
            rejected.append(ident)
    raise CheckpointError("no valid resumable checkpoint remains")


def restore_workspace(archive, metadata, state, workspace):
    directory = archive / "checkpoints" / metadata["checkpoint_id"]
    _no_link(workspace)
    _no_link(workspace.parent)
    workspace.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".restore-", dir=workspace.parent))
    try:
        with zipfile.ZipFile(directory / FILES["workspace"]) as bundle:
            seen = set()
            for entry in bundle.infolist():
                name = entry.orig_filename
                relative = _member_path(name)
                if name in seen:
                    raise CheckpointError("unsafe workspace snapshot path")
                seen.add(name)
                target = stage.joinpath(*relative.parts)
                if not target.resolve().is_relative_to(stage.resolve()):
                    raise CheckpointError("workspace snapshot path escaped staging")
                if entry.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with bundle.open(entry) as source, target.open("xb") as dest:
                        shutil.copyfileobj(source, dest)
        if state["memory_present"]:
            shutil.copyfile(directory / FILES["memory"], stage / ".memory.sqlite")
        # Preserve the former workspace, including any operator edits since pause.
        if workspace.exists():
            backup = workspace.with_name(workspace.name + ".before-resume-" + stage.name[9:])
            boundary = workspace.parent.resolve()
            if any(not p.resolve().is_relative_to(boundary) for p in (workspace, backup, stage)):
                raise CheckpointError("workspace move escaped its data-root directory")
            os.replace(workspace, backup)
        os.replace(stage, workspace)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        if stage.exists():
            _remove_owned_tree(stage, workspace.parent)
        raise CheckpointError(f"workspace restore failed ({type(error).__name__})") from None
