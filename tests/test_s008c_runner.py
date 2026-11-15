"""The statistics check must not spend before clearance or after its stop/cap."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from open_lab_simulation.policies import digest


@pytest.fixture
def runner(tmp_path, monkeypatch):
    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("s008c_runner", repo / "scripts/s008c_check.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "data_root", lambda: tmp_path)
    monkeypatch.setattr(sys, "argv", ["s008c_check.py"])

    def refused(*args, **kwargs):
        raise AssertionError("a run must not start")

    monkeypatch.setattr(module, "run_experiment", refused)
    manifest = json.loads((repo / "campaigns/s008c-001.json").read_text(encoding="utf-8"))
    return module, manifest


def progress(tmp_path, manifest, started, outcomes, **extra):
    directory = tmp_path / "capability-checks/s008c-001"
    directory.mkdir(parents=True)
    path = directory / "progress.json"
    path.write_text(json.dumps({"manifest_sha256": digest(manifest),
                                "started": started, "outcomes": outcomes, **extra}),
                    encoding="utf-8")
    return path


def test_no_clearance_starts_no_run(runner, tmp_path):
    module, manifest = runner
    with pytest.raises(SystemExit, match="quota confirmation required"):
        module.main()
    assert not (tmp_path / "capability-checks/s008c-001/progress.json").exists()


def test_clearance_cannot_bypass_provider_stop(runner, tmp_path, monkeypatch):
    module, manifest = runner
    path = progress(tmp_path, manifest, ["guided"], [{"stop": True}])
    before = path.read_bytes()
    monkeypatch.setattr(sys, "argv", ["s008c_check.py", "--quota-clearance", "test confirmation"])
    with pytest.raises(SystemExit, match="stopped the check"):
        module.main()
    assert path.read_bytes() == before


def test_four_started_cells_exhaust_the_allowance(runner, tmp_path):
    module, manifest = runner
    progress(tmp_path, manifest, [c["id"] for c in manifest["cells"]],
             [{"stop": False}] * 4, quota_clearance="test confirmation")
    with pytest.raises(SystemExit, match="All four authorized cells already started"):
        module.main()


def test_failed_fake_run_consumes_one_cell_and_stops(runner, tmp_path, monkeypatch):
    module, manifest = runner
    archive = tmp_path / "fake-archive"
    archive.mkdir()
    (archive / "events.jsonl").write_text("{}\n", encoding="utf-8")
    (archive / "provider-ledger.jsonl").write_text(
        json.dumps({"kind": "outcome", "result": "terminal"}) + "\n", encoding="utf-8")
    received = []

    def fake_run(path):
        received.append(path.name)
        return SimpleNamespace(archive=archive, run_id="fake-quota", state="failed", reason="quota")

    monkeypatch.setattr(module, "run_experiment", fake_run)
    monkeypatch.setattr(module, "verify_archive", lambda path: [])
    monkeypatch.setattr(sys, "argv", ["s008c_check.py", "--quota-clearance", "test confirmation"])
    module.main()
    record = json.loads((tmp_path / "capability-checks/s008c-001/progress.json").read_text())
    assert record["quota_clearance"] == "test confirmation"
    assert record["started"] == ["guided"]
    assert record["outcomes"][0]["stop"] is True
    with pytest.raises(SystemExit, match="stopped the check"):
        module.main()
    assert received == ["guided.json"]
