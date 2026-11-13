"""Locate a world release and refuse it unless it is exactly what its manifest says.

An independent implementation of the content digest specified in the
Builder's ``docs/architecture/CONTRACTS.md`` ("World release"). No Builder
code is imported.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .contracts import Contracts, canonical_json, sha256_text

MANIFEST_NAME = "world-release.json"
VOLATILE_FIELDS = ("created_at", "builder", "content_digest")


class WorldError(Exception):
    """The world release is missing, malformed or not what its digest says."""


@dataclass
class World:
    directory: Path
    manifest: dict[str, Any]
    contracts: Contracts
    # capability id -> (directory, manifest)
    capabilities: dict[str, tuple[Path, dict[str, Any]]] = field(default_factory=dict)

    @property
    def digest(self) -> str:
        return self.manifest["content_digest"]["value"]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def compute_content_digest(release_dir: Path, manifest: dict[str, Any]) -> str:
    """Recompute a release's content digest from the files on disk."""
    lines = []
    for path in release_dir.rglob("*"):
        if path.is_symlink():
            raise WorldError("the release contains a symbolic link")
        relative = path.relative_to(release_dir).as_posix()
        if path.is_file() and relative != MANIFEST_NAME:
            lines.append((relative.encode("utf-8"), f"{file_sha256(path)}  {relative}\n"))
    stable = {key: value for key, value in manifest.items() if key not in VOLATILE_FIELDS}
    text = "".join(line for _, line in sorted(lines))
    text += f"{sha256_text(canonical_json(stable))}  {MANIFEST_NAME}\n"
    return sha256_text(text)


def load_world(release_dir: Path, expected_digest: str | None = None) -> World:
    """Verify a release and load its capability manifests. Raises ``WorldError``."""
    try:
        manifest = json.loads((release_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
        contracts = Contracts(release_dir / "contracts")
    except (OSError, ValueError, KeyError) as error:
        raise WorldError(f"cannot read the world release ({type(error).__name__})") from None

    problems = contracts.validate(manifest, "world-release")
    if problems:
        raise WorldError(f"{MANIFEST_NAME} is invalid: {problems[0]}")
    if manifest["status"] not in ("validated", "released"):
        raise WorldError(f"world status is '{manifest['status']}'; only built releases can be run")
    actual = compute_content_digest(release_dir, manifest)
    if actual != manifest["content_digest"]["value"]:
        raise WorldError("the world release does not match its content digest")
    if expected_digest is not None and actual != expected_digest:
        raise WorldError("the world release is not the one the experiment names (digest differs)")

    world = World(release_dir, manifest, contracts)
    for entry in manifest["capabilities"]:
        path = release_dir / entry["manifest_path"]
        document = json.loads(path.read_text(encoding="utf-8"))
        problems = contracts.validate(document, "capability")
        if problems or document["id"] != entry["id"]:
            raise WorldError(f"capability '{entry['id']}' has an invalid manifest")
        world.capabilities[entry["id"]] = (path.parent, document)
    return world
