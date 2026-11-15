from __future__ import annotations

import hashlib
import json
import shutil
import socket
import sys
from pathlib import Path

import pytest

from conftest import call
from open_lab_simulation import evidence
from open_lab_simulation.cli import main
from open_lab_simulation.contracts import canonical_json, jsonl_lines
from open_lab_simulation.policies import digest
from open_lab_simulation.supervisor import run_experiment
from open_lab_simulation.world import file_sha256

SCHEMAS = Path(__file__).parent / "fixtures/research-contracts"


def save(path, document):
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@pytest.fixture
def study(root, make_config, tmp_path, monkeypatch):
    directory = tmp_path / "study"
    directory.mkdir()
    cells = []
    for name, replies in (
        ("positive", [{"tool_calls": [call("file.write", path="note.md", content="a\u2028b")]}]),
        ("idle", [{"text": "No action selected."}]),
        ("failed", []),
        ("unstarted", []),
        ("refused", []),
    ):
        config_path = make_config(replies, name=name)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        saved = directory / f"{name}.json"
        shutil.copyfile(config_path, saved)
        outcome = {"status": "not_started" if name == "unstarted" else
                   "launch_failed" if name == "refused" else "recorded",
                   "archive": None, "detail": "Synthetic engineering fixture; no real model."}
        if outcome["status"] == "recorded":
            if name == "failed":
                # A trusted fake runtime ends with the ordinary model-error reason.
                code = ('import sys,json;sys.stdin.readline();'
                        'print(json.dumps({"end":{"reason":"model_error","steps":0}}),flush=True)')
                with monkeypatch.context() as patch:
                    patch.setattr("open_lab_simulation.supervisor.WORKER_COMMAND",
                                  [sys.executable, "-c", code])
                    result = run_experiment(config_path, root)
                # This is a synthetic provider refusal, never a real model request.
                (result.archive / "provider-ledger.jsonl").write_text(json.dumps({
                    "kind": "outcome", "result": "terminal",
                    "code": "fixture_subscription_sharing_usage_limit_exceeded"}) + "\n",
                    encoding="utf-8")
            else:
                result = run_experiment(config_path, root)
            outcome["archive"] = {"experiment_id": name, "run_id": result.run_id}
        cells.append({"cell_id": name, "role": "pilot" if name == "idle" else "engineering",
                      "condition": name, "replicate": 1, "seed_label": None,
                      "configuration": {"path": saved.name, "sha256": digest(config)},
                      "outcome": outcome})
    attachment = directory / "analysis.md"
    attachment.write_text("Observations are not agent motivation.\n", encoding="utf-8")
    spec = {"schema": "open-lab/study-manifest", "schema_version": "1.0.0",
            "study_id": "test-evidence", "question": "Do all declared cells survive export?",
            "registration": {"kind": "engineering", "commit": None,
                             "statement": "NOT_SCORED fixture, no prospective claim."},
            "comparison": {"changed_fields": ["/extensions/scripted_model"],
                           "constant_fields": ["/permissions", "/unknown"],
                           "seed_interpretation": "Not applicable.",
                           "analysis_policy": "Do not infer motivation or success.",
                           "limitations": ["Synthetic fixtures only."]},
            "cells": cells, "attachments": [{"kind": "analysis_plan", "path": "analysis.md",
                                               "sha256": file_sha256(attachment)}]}
    path = directory / "study.json"
    save(path, spec)
    return path, spec


def source_archive(root, spec, name="positive"):
    ref = next(c for c in spec["cells"] if c["cell_id"] == name)["outcome"]["archive"]
    return root / "runs" / ref["experiment_id"] / ref["run_id"]


def test_export_roundtrip_denominators_determinism_and_relocation(
    study, root, tmp_path, monkeypatch,
):
    path, spec = study
    before = evidence._snapshot(root)
    one, two = tmp_path / "one", tmp_path / "two"

    def no_network(*args, **kwargs):
        raise AssertionError("offline export must not make a network request")

    monkeypatch.setattr(socket, "create_connection", no_network)
    first = evidence.export_study(path, SCHEMAS, root, one)
    second = evidence.export_study(path, SCHEMAS, root, two)
    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert before == evidence._snapshot(root)
    manifest = json.loads((one / "export.json").read_text(encoding="utf-8"))
    assert [r["cell_id"] for r in manifest["cells"]] == [c["cell_id"] for c in spec["cells"]]
    assert [r["declared_status"] for r in manifest["cells"]] == [
        "recorded", "recorded", "recorded", "not_started", "launch_failed"]
    assert [r["record"]["final_state"] for r in manifest["cells"][:3]] == [
        "completed", "completed", "failed"]
    assert manifest["cells"][2]["record"]["provider_terminal_codes"] == [
        "fixture_subscription_sharing_usage_limit_exceeded"]
    assert all(r["record"]["model"]["weight_digest"] is None for r in manifest["cells"][:3])
    assert any(r["role"] == "pilot" for r in manifest["cells"])
    assert any(c["status"] == "missing" for c in manifest["constant_checks"])
    assert any(c["status"] == "equal" for c in manifest["constant_checks"])
    assert any(o["path"].endswith("runtime.stderr.log") for o in manifest["omissions"])
    assert not list((one / "runs").rglob("runtime.stderr.log"))
    assert not list((one / "runs").rglob(".writer.guard"))
    # Relocation and loss of all original inputs/root cannot affect verification.
    relocated = tmp_path / "relocated"
    shutil.copytree(one, relocated)
    root.rename(tmp_path / "hidden-root")
    path.parent.rename(tmp_path / "hidden-inputs")
    checked = evidence.verify_export(relocated, first["manifest_sha256"])
    assert checked["verified"] and checked["external_pin_checked"]
    assert checked["cells"] == 5 and checked["archives"] == 3


@pytest.mark.parametrize("change", ["events", "extra", "manifest", "schema"])
def test_tampering_is_refused(study, root, tmp_path, change):
    path, spec = study
    dest = tmp_path / "export"
    result = evidence.export_study(path, SCHEMAS, root, dest)
    if change == "events":
        target = next((dest / "runs").rglob("events.jsonl"))
        target.write_bytes(target.read_bytes() + b"{}\n")
    elif change == "extra":
        (dest / "extra.txt").write_text("unindexed", encoding="utf-8")
    elif change == "manifest":
        doc = json.loads((dest / "export.json").read_text(encoding="utf-8"))
        doc["cells"].pop()
        save(dest / "export.json", doc)
    else:
        target = dest / "contracts/v1/study-manifest.schema.json"
        target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(evidence.EvidenceError):
        evidence.verify_export(dest, result["manifest_sha256"])


@pytest.mark.parametrize("change", ["cell", "archive", "pin", "attachment", "active", "damaged"])
def test_invalid_sources_refused_without_publication(study, root, tmp_path, change):
    path, spec = study
    if change == "cell":
        spec["cells"][1]["cell_id"] = spec["cells"][0]["cell_id"]
    elif change == "archive":
        spec["cells"][1]["outcome"]["archive"] = spec["cells"][0]["outcome"]["archive"]
    elif change == "pin":
        spec["cells"][0]["configuration"]["sha256"] = "0" * 64
    elif change == "attachment":
        (path.parent / "analysis.md").write_text("changed", encoding="utf-8")
    elif change == "active":
        (source_archive(root, spec) / ".writer.lock").write_text("active", encoding="utf-8")
    else:
        (source_archive(root, spec) / "events.jsonl").write_text("{}\n", encoding="utf-8")
    save(path, spec)
    dest = tmp_path / "export"
    with pytest.raises((evidence.EvidenceError, ValueError)):
        evidence.export_study(path, SCHEMAS, root, dest)
    assert not dest.exists()


def test_changing_source_retains_failed_stage_and_never_changes_destination(
    study, root, tmp_path, monkeypatch,
):
    path, spec = study
    real_copy = shutil.copyfile
    changed = False
    archive = source_archive(root, spec)

    def mutate(source, target, *args, **kwargs):
        nonlocal changed
        result = real_copy(source, target, *args, **kwargs)
        if not changed:
            changed = True
            (archive / "injected.txt").write_text("concurrent change", encoding="utf-8")
        return result

    monkeypatch.setattr(evidence.shutil, "copyfile", mutate)
    dest = tmp_path / "export"
    with pytest.raises(evidence.EvidenceError, match="changed during export"):
        evidence.export_study(path, SCHEMAS, root, dest)
    assert not dest.exists()
    assert list(tmp_path.glob(".export.staging-*"))


def test_output_collisions_unsafe_paths_and_links_are_refused(study, root, tmp_path, monkeypatch):
    path, spec = study
    destination = tmp_path / "existing"
    destination.mkdir()
    with pytest.raises(evidence.EvidenceError, match="already exists"):
        evidence.export_study(path, SCHEMAS, root, destination)
    with pytest.raises(evidence.EvidenceError, match="only be written under exports"):
        evidence.export_study(path, SCHEMAS, root, root / "runs/forbidden")
    for value in ("../escape", "a\\b", "a/CON.txt", "a/file:stream", "a/../b", "a?.txt"):
        with pytest.raises(evidence.EvidenceError):
            evidence._path(value)
    linked = source_archive(root, spec) / "events.jsonl"
    original = evidence.is_link
    monkeypatch.setattr(evidence, "is_link", lambda p: p == linked or original(p))
    with pytest.raises(evidence.EvidenceError, match="links or junctions"):
        evidence.export_study(path, SCHEMAS, root, tmp_path / "linked")


def test_config_mismatch_is_explicit_not_silently_matched(study, root, tmp_path):
    path, spec = study
    cell = spec["cells"][0]
    config_path = path.parent / cell["configuration"]["path"]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["initial_prompt"] = "A different frozen condition"
    save(config_path, config)
    cell["configuration"]["sha256"] = digest(config)
    save(path, spec)
    dest = tmp_path / "export"
    result = evidence.export_study(path, SCHEMAS, root, dest)
    assert result["inventory_findings"] == 1
    doc = json.loads((dest / "export.json").read_text(encoding="utf-8"))
    assert doc["cells"][0]["findings"] == ["CONFIGURATION_MISMATCH"]


def test_cli_export_verify_and_wrong_external_pin(study, root, tmp_path, capsys):
    path, spec = study
    dest = tmp_path / "cli-export"
    assert main(["evidence", "export", str(path), "--schemas", str(SCHEMAS),
                 "--data-root", str(root), "--out", str(dest)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert main(["evidence", "verify", str(dest), "--sha256", result["manifest_sha256"]]) == 0
    assert main(["evidence", "verify", str(dest), "--sha256", "0" * 64]) == 2


def test_case_collision_and_unsealed_source_are_refused(study, root, tmp_path):
    path, spec = study
    spec["attachments"].append({**spec["attachments"][0], "path": "ANALYSIS.md"})
    save(path, spec)
    with pytest.raises(evidence.EvidenceError, match="case-colliding"):
        evidence.export_study(path, SCHEMAS, root, tmp_path / "collision")
    spec["attachments"].pop()
    save(path, spec)
    (source_archive(root, spec) / "summary.json").unlink()
    with pytest.raises(OSError):
        evidence.export_study(path, SCHEMAS, root, tmp_path / "unsealed")
    assert not (tmp_path / "unsealed").exists()


@pytest.mark.parametrize("initial,expected", [
    (None, "unknown"),
    ({"files": [], "sha256": "0" * 64, "memory_present": False}, "unknown"),
    ({"files": ["note.md"], "sha256": digest(["note.md"]), "memory_present": False}, "nonempty"),
])
def test_initial_state_unknown_and_nonempty_are_not_invented_as_fresh(
    study, root, tmp_path, initial, expected,
):
    path, spec = study
    archive = source_archive(root, spec)
    events = [json.loads(line) for line in jsonl_lines(
        (archive / "events.jsonl").read_text(encoding="utf-8"))]
    # Generate an internally valid legacy/alternate instrumentation fixture.
    # This edits only a synthetic test archive, never real recorded evidence.
    extension = events[0]["payload"]["extensions"]
    if initial is None:
        extension.pop("initial_workspace")
    else:
        extension["initial_workspace"] = initial
    lines = []
    for event in events:
        if lines:
            event["prev_event_sha256"] = hashlib.sha256(lines[-1].encode()).hexdigest()
        lines.append(canonical_json(event))
    (archive / "events.jsonl").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    summary = json.loads((archive / "summary.json").read_text(encoding="utf-8"))
    summary["last_event_sha256"] = hashlib.sha256(lines[-1].encode()).hexdigest()
    save(archive / "summary.json", summary)
    dest = tmp_path / "export"
    evidence.export_study(path, SCHEMAS, root, dest)
    exported = json.loads((dest / "export.json").read_text(encoding="utf-8"))["cells"][0]
    assert exported["record"]["initial_state"] == expected
    assert f"INITIAL_STATE_{expected.upper()}" in exported["findings"]


def test_requested_constants_are_checked_by_role_without_pooling(study, root, tmp_path):
    path, spec = study
    spec["comparison"]["constant_fields"] = ["/limits"]
    cell = next(c for c in spec["cells"] if c["cell_id"] == "unstarted")
    config_path = path.parent / cell["configuration"]["path"]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["limits"]["wall_clock_s"] = 2
    save(config_path, config)
    cell["configuration"]["sha256"] = digest(config)
    save(path, spec)
    dest = tmp_path / "export"
    evidence.export_study(path, SCHEMAS, root, dest)
    checks = json.loads((dest / "export.json").read_text(encoding="utf-8"))["constant_checks"]
    assert next(c for c in checks if c["role"] == "engineering")["status"] == "different"
    assert next(c for c in checks if c["role"] == "pilot")["status"] == "equal"
