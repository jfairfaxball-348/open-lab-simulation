from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest

from open_lab_simulation.world import MANIFEST_NAME, compute_content_digest, file_sha256

REPO = Path(__file__).resolve().parent.parent
# A release built by the Builder (olb world build worlds/minimal), copied here as test data.
FIXTURE_WORLD = REPO / "tests" / "fixtures" / "worlds" / "minimal" / "0.4.0"

# Capabilities that exist only for these tests. Each is {id: (permissions, timeout, code)}.
TEST_CAPABILITIES = {
    "file.write": ({"filesystem": "workspace_write", "network": "none"}, 10, '''
import os
from pathlib import Path

def run(arguments):
    target = Path(os.environ["OPEN_LAB_WORKSPACE"]) / arguments["path"]
    if arguments.get("content") is None:
        target.unlink()
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(arguments["content"], encoding="utf-8")
    return {"ok": True}
'''),
    "test.heartbeat": ({"filesystem": "workspace_write", "network": "none"}, 120, '''
import os, time
from pathlib import Path

def run(arguments):
    target = Path(os.environ["OPEN_LAB_WORKSPACE"]) / "heartbeat.txt"
    for beat in range(100000):
        target.write_text(str(beat))
        time.sleep(0.05)
    return {"ok": True}
'''),
    "test.environment": ({"filesystem": "none", "network": "none"}, 10, '''
import os

def run(arguments):
    return {"ok": True, "names": sorted(os.environ)}
'''),
    "test.outside": ({"filesystem": "none", "network": "none"}, 10, '''
import os

def run(arguments):
    return {"ok": "OPEN_LAB_WORKSPACE" not in os.environ}
'''),
}


def restamp(release: Path, change=None) -> dict:
    """Rewrite a release's manifest so it is consistent with the files now on disk."""
    path = release / MANIFEST_NAME
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if change:
        change(manifest)
    files = sorted(
        p.relative_to(release).as_posix()
        for p in release.rglob("*") if p.is_file() and p.name != MANIFEST_NAME
    )
    manifest["extensions"] = {
        "files": [{"path": f, "sha256": file_sha256(release / f)} for f in files]
    }
    manifest["content_digest"]["value"] = compute_content_digest(release, manifest)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    return manifest


def add_test_capabilities(release: Path) -> None:
    template = json.loads(
        (release / "capabilities" / "text.stats" / "capability.json").read_text(encoding="utf-8")
    )
    entries = []
    for capability_id, (permissions, timeout_s, code) in TEST_CAPABILITIES.items():
        directory = release / "capabilities" / capability_id
        directory.mkdir()
        (directory / "impl.py").write_text(code, encoding="utf-8", newline="\n")
        manifest = copy.deepcopy(template)
        manifest.update(
            id=capability_id,
            name=capability_id,
            invocation={"interface": "python", "entrypoint": "impl:run"},
            input_schema={"type": "object"},
            output_schema={"type": "object", "required": ["ok"]},
            permissions=permissions,
            resources={"timeout_s": timeout_s},
            demonstrations=[{"id": "none", "title": "Not run", "input": {},
                             "deterministic": False}],
            self_test={"demonstrations": ["none"]},
        )
        (directory / "capability.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        entries.append({
            "id": capability_id, "version": manifest["version"],
            "manifest_path": f"capabilities/{capability_id}/capability.json",
            "sha256": file_sha256(directory / "capability.json"),
        })
    restamp(release, lambda m: m["capabilities"].extend(entries))


@pytest.fixture
def root(tmp_path) -> Path:
    """A private data root holding the minimal world plus the test-only capabilities."""
    data = tmp_path / "data"
    release = data / "worlds" / "minimal" / "0.4.0"
    shutil.copytree(FIXTURE_WORLD, release)
    add_test_capabilities(release)
    return data


@pytest.fixture
def release(root) -> Path:
    return root / "worlds" / "minimal" / "0.4.0"


def call(capability_id: str, **arguments) -> dict:
    return {"capability_id": capability_id, "arguments": arguments}


@pytest.fixture
def make_config(tmp_path):
    """Write an experiment configuration and return its path."""
    def make(replies, *, wall_clock_s=60, name="test-run", **changes) -> Path:
        config = {
            "schema": "open-lab/experiment-config",
            "schema_version": "1.1.0",
            "experiment_id": name,
            "title": "Test run",
            "created_at": "2026-11-13T00:00:00Z",
            "initial_prompt": "You may use whatever you find.",
            "world": {"world_id": "minimal", "version": "0.4.0"},
            "orchestrator": {"provider_id": "scripted", "model_id": "scripted"},
            "limits": {"wall_clock_s": wall_clock_s},
            "permissions": {"filesystem": "workspace_write", "network": "none"},
            "termination": {"on_deadline": "hard_stop"},
            "extensions": {"scripted_model": {"replies": replies}},
        }
        for key, value in changes.items():
            if isinstance(value, dict) and isinstance(config.get(key), dict):
                config[key] = {**config[key], **value}
            else:
                config[key] = value
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        return path
    return make


def events_of(archive: Path) -> list[dict]:
    return [json.loads(line) for line in
            (archive / "events.jsonl").read_text(encoding="utf-8").splitlines()]


def tree(directory: Path) -> set[str]:
    return {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
