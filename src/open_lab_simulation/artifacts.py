"""Find what appeared in the workspace and preserve it in the run archive.

Artifacts are discovered by the supervisor scanning the workspace; the agent
never announces them. Every version of every file is copied into a
content-addressed store in the archive, so the record survives the workspace.
"""

from __future__ import annotations

import hashlib
import json
import mimetypes
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import Contracts, canonical_json, jsonl_lines
from .paths import walk_files
from .recorder import Recorder

# Larger files are recorded by digest only.
MAX_PRESERVED_BYTES = 50 * 1024 * 1024


@dataclass
class _Known:
    artifact_id: str
    sha256: str
    revision: int
    stamp: tuple[int, int]  # size, mtime_ns: a cheap "probably unchanged" check


class WorkspaceScanner:
    def __init__(
        self,
        workspace: Path,
        archive: Path,
        recorder: Recorder,
        contracts: Contracts,
        actor: dict[str, Any],
        max_preserved_bytes: int = MAX_PRESERVED_BYTES,
        resume: bool = False,
    ):
        self.workspace = workspace
        self.archive = archive
        self.recorder = recorder
        self.contracts = contracts
        self.actor = actor
        self.max_preserved_bytes = max_preserved_bytes
        self._known: dict[str, _Known] = {}
        self._next_id = 1
        self.versions = 0
        (archive / "artifacts").mkdir(exist_ok=True)
        if resume:
            metadata = read_index(archive)
            self.versions = len(metadata)
            self._next_id = (
                max((int(m["artifact_id"].split("-")[-1]) for m in metadata), default=0) + 1
            )
            records = {m["artifact_id"]: m for m in metadata}
            for line in jsonl_lines((archive / "events.jsonl").read_text(encoding="utf-8")):
                event = json.loads(line)
                payload = event["payload"]
                if event["type"] in ("artifact.created", "artifact.modified"):
                    m = records[payload["artifact_id"]]
                    self._known[payload["path"]] = _Known(
                        m["artifact_id"], m["sha256"], m["revision"], (-1, -1)
                    )
                elif event["type"] == "artifact.deleted":
                    self._known.pop(payload["path"], None)

    def _files(self) -> dict[str, Path]:
        # A link could point anywhere on the host; never follow or preserve one.
        return {path.relative_to(self.workspace).as_posix(): path
                for path in walk_files(self.workspace)}

    def _preserve(self, path: Path) -> tuple[str, int, bool] | None:
        """Hash a file and copy it into the store. None if it cannot be read right now."""
        temporary = self.archive / "artifacts" / ".incoming"
        digest = hashlib.sha256()
        size = 0
        try:
            with path.open("rb") as source, temporary.open("wb") as target:
                for block in iter(lambda: source.read(1 << 20), b""):
                    digest.update(block)
                    size += len(block)
                    if size <= self.max_preserved_bytes:
                        target.write(block)
        except OSError:
            temporary.unlink(missing_ok=True)
            return None
        value = digest.hexdigest()
        preserved = size <= self.max_preserved_bytes
        stored = self.archive / "artifacts" / "sha256" / value[:2] / value
        if preserved and not stored.exists():
            stored.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(temporary, stored)
        temporary.unlink(missing_ok=True)
        return value, size, preserved

    def scan(self, producer: dict[str, Any]) -> int:
        """Record every change since the last scan. Returns the number of events written."""
        events = 0
        files = self._files()
        for relative in sorted(files):
            path = files[relative]
            try:
                status = path.stat()
            except OSError:
                continue
            stamp = (status.st_size, status.st_mtime_ns)
            known = self._known.get(relative)
            if known and known.stamp == stamp:
                continue
            preserved = self._preserve(path)
            if preserved is None:
                continue
            sha256, size, kept = preserved
            if known and known.sha256 == sha256:
                known.stamp = stamp
                continue

            artifact_id = f"art-{self._next_id:06d}"
            self._next_id += 1
            event = self.recorder.record(
                recorder="supervisor",
                type="artifact.modified" if known else "artifact.created",
                actor=self.actor,
                payload={
                    "artifact_id": artifact_id,
                    "path": relative,
                    "sha256": sha256,
                    "size_bytes": size,
                },
            )
            metadata: dict[str, Any] = {
                "schema": "open-lab/artifact-metadata",
                "schema_version": self.contracts.version("artifact-metadata"),
                "artifact_id": artifact_id,
                "experiment_id": event["experiment_id"],
                "run_id": event["run_id"],
                "workspace_path": relative,
                "preserved": kept,
                "sha256": sha256,
                "size_bytes": size,
                "observed_at": event["timestamp"],
                "observed_at_seq": event["seq"],
                "revision": known.revision + 1 if known else 1,
                "producer": producer,
            }
            if kept:
                metadata["archive_path"] = f"artifacts/sha256/{sha256[:2]}/{sha256}"
            if known:
                metadata["supersedes"] = known.artifact_id
            media_type = mimetypes.guess_type(relative)[0]
            if media_type:
                metadata["media_type"] = media_type
            problems = self.contracts.validate(metadata, "artifact-metadata")
            if problems:
                raise ValueError(f"artifact metadata invalid: {problems[0]}")
            with (self.archive / "artifacts" / "index.jsonl").open(
                "a", encoding="utf-8", newline="\n"
            ) as index:
                index.write(canonical_json(metadata) + "\n")
            self._known[relative] = _Known(artifact_id, sha256, metadata["revision"], stamp)
            self.versions += 1
            events += 1

        for relative in sorted(self._known.keys() - files.keys()):
            known = self._known.pop(relative)
            self.recorder.record(
                recorder="supervisor",
                type="artifact.deleted",
                actor=self.actor,
                payload={"artifact_id": known.artifact_id, "path": relative},
            )
            events += 1
        return events


def read_index(archive: Path) -> list[dict[str, Any]]:
    index = archive / "artifacts" / "index.jsonl"
    if not index.is_file():
        return []
    return [json.loads(line) for line in jsonl_lines(index.read_text(encoding="utf-8")) if line]
