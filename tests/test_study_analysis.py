from __future__ import annotations

import copy
import json
import shutil
import socket

import pytest

import test_evidence
from open_lab_simulation import study_analysis as analysis
from open_lab_simulation.cli import main
from open_lab_simulation.evidence import EvidenceError, _snapshot, export_study
from test_evidence import SCHEMAS, save

study = test_evidence.study


@pytest.fixture
def exported(study, root, tmp_path):
    path, spec = study
    directory = tmp_path / "export"
    export_study(path, SCHEMAS, root, directory)
    return directory


def annotations(report):
    doc = copy.deepcopy(report["annotations"])
    doc["reviewer"] = {"id": "fixture-reviewer", "masking": "unmasked"}
    doc["limitations"] = ["Synthetic single-rater fixture, not scientific validation."]
    positive = report["cells"][0]
    artifact = positive["trace"]["artifacts"][0]
    ref = {
        "cell_id": positive["cell_id"],
        "event_sha256": positive["trace"]["event_sha256"],
        "seqs": [artifact["observed_at_seq"]],
        "artifacts": [artifact["sha256"]],
    }
    doc["entries"] = [
        {
            "id": "artifact",
            "kind": "assessment",
            "category": "preservation",
            "label": "supported",
            "rationale": "A preserved revision is recorded.",
            "evidence": [ref],
            "quote": None,
            "quote_pointer": None,
        }
    ]
    return doc


def test_reproducible_relocated_read_only_roles_and_denominators(exported, tmp_path, monkeypatch):
    before = _snapshot(exported)

    def no_network(*args, **kwargs):
        raise AssertionError("offline analysis may not request a model or network")

    monkeypatch.setattr(socket, "create_connection", no_network)
    one = tmp_path / "analysis-one"
    two = tmp_path / "analysis-two"
    a = analysis.write_analysis(exported, SCHEMAS, one)
    b = analysis.write_analysis(exported, SCHEMAS, two)
    assert a["analysis_sha256"] == b["analysis_sha256"]
    assert before == _snapshot(exported)
    report = json.loads((one / "analysis.json").read_text(encoding="utf-8"))
    assert len(report["cells"]) == 5
    assert sum(g["planned"] for g in report["groups"]) == 5
    assert [c["declared_status"] for c in report["cells"]][-2:] == ["not_started", "launch_failed"]
    assert report["cells"][2]["trace"]["provider_censored"]
    assert report["cells"][2]["record"]["final_state"] == "failed"
    assert report["cells"][3]["trace"] is None
    pilot = next(g for g in report["groups"] if g["role"] == "pilot")
    assert pilot["cells"] == ["idle"]
    assert all("idle" not in c["cells"] for c in report["contrasts"])
    assert all(c["causal_match"] == "not_established" for c in report["contrasts"])
    assert all(c["independence"] == "not_established" for c in report["contrasts"])
    assert any(c["configuration_differences"] for c in report["contrasts"])
    assert report["cells"][0]["trace"]["model"]["weight_digest"] is None
    assert report["cells"][0]["trace"]["artifacts"][0]["observed_at_seq"] >= 0
    assert [g["id"] for g in report["validity_gates"]] == list(range(1, 8))
    assert all(g["human_or_unverified"] for g in report["validity_gates"])
    assert all(g["mechanical_status"] == "not_checked" for g in report["validity_gates"][-2:])
    relocated = tmp_path / "relocated-export"
    shutil.copytree(exported, relocated)
    exported.rename(tmp_path / "hidden-export")
    one.rename(tmp_path / "relocated-report")
    assert analysis.verify_analysis(relocated, tmp_path / "relocated-report", a["analysis_sha256"])[
        "verified"
    ]


def test_cited_assessments_exact_quotes_and_disagreements_stay_separate(exported, tmp_path):
    report = analysis.analyze(exported, SCHEMAS)
    doc = annotations(report)
    idle = report["cells"][1]
    ref = {
        "cell_id": idle["cell_id"],
        "event_sha256": idle["trace"]["event_sha256"],
        "seqs": idle["trace"]["seqs"]["agent_outputs"][:1],
        "artifacts": [],
    }
    doc["entries"].append(
        {
            "id": "quote",
            "kind": "agent_claim",
            "category": "statement",
            "label": "uncertain",
            "rationale": "Assertion only, not internal state.",
            "evidence": [ref],
            "quote": "No action selected.",
            "quote_pointer": "/payload/text",
        }
    )
    doc["entries"].append(
        {
            **copy.deepcopy(doc["entries"][0]),
            "id": "disagreement",
            "label": "uncertain",
            "rationale": "Preservation alone does not show result use.",
        }
    )
    path = tmp_path / "inputs/annotations.json"
    path.parent.mkdir()
    save(path, doc)
    target = tmp_path / "reviewed"
    result = analysis.write_analysis(exported, SCHEMAS, target, path)
    assert result["annotations"] == 3
    actual = json.loads((target / "analysis.json").read_text(encoding="utf-8"))
    assert actual["annotations"] == doc
    assert actual["annotations"]["reviewer"]["masking"] == "unmasked"


@pytest.mark.parametrize(
    "change",
    [
        "duplicate",
        "cell",
        "missing",
        "hash",
        "seq",
        "artifact",
        "quote",
        "pointer",
        "kind",
        "export",
    ],
)
def test_invalid_annotation_citations_refuse_publication(exported, tmp_path, change):
    report = analysis.analyze(exported, SCHEMAS)
    doc = annotations(report)
    entry = doc["entries"][0]
    ref = entry["evidence"][0]
    if change == "duplicate":
        doc["entries"].append(copy.deepcopy(entry))
    elif change == "cell":
        ref["cell_id"] = "unknown"
    elif change == "missing":
        ref["cell_id"] = "unstarted"
    elif change == "hash":
        ref["event_sha256"] = "0" * 64
    elif change == "seq":
        ref["seqs"] = [999999]
    elif change == "artifact":
        ref["artifacts"] = ["0" * 64]
    elif change == "export":
        doc["export_sha256"] = "0" * 64
    else:
        entry.update(kind="agent_claim", quote="fabricated quote", quote_pointer="/payload/text")
        if change == "pointer":
            entry["quote_pointer"] = "/timestamp"
        elif change == "kind":
            entry["kind"] = "observation"
    path = tmp_path / "input/annotation.json"
    path.parent.mkdir()
    save(path, doc)
    target = tmp_path / "refused"
    with pytest.raises(EvidenceError):
        analysis.write_analysis(exported, SCHEMAS, target, path)
    assert not target.exists()


@pytest.mark.parametrize("change", ["analysis", "markdown", "extra", "schema", "annotation"])
def test_report_tampering_refused_without_relying_on_external_pin(exported, tmp_path, change):
    out = tmp_path / "analysis"
    analysis.write_analysis(exported, SCHEMAS, out)
    if change == "analysis":
        doc = json.loads((out / "analysis.json").read_text(encoding="utf-8"))
        doc["groups"][0]["planned"] += 1
        save(out / "analysis.json", doc)
    elif change == "markdown":
        (out / "report.md").write_text("invented success", encoding="utf-8")
    elif change == "extra":
        (out / "extra.txt").write_text("extra", encoding="utf-8")
    elif change == "schema":
        path = out / "contracts/v1/study-analysis.schema.json"
        path.write_bytes(path.read_bytes() + b" ")
    else:
        doc = json.loads((out / "annotations.json").read_text(encoding="utf-8"))
        doc["reviewer"]["id"] = "replacement"
        save(out / "annotations.json", doc)
    with pytest.raises(EvidenceError):
        analysis.verify_analysis(exported, out)


def test_source_changes_invalid_export_and_output_collisions_refused(
    exported, tmp_path, monkeypatch
):
    out = tmp_path / "analysis"
    out.mkdir()
    with pytest.raises(EvidenceError, match="new and outside"):
        analysis.write_analysis(exported, SCHEMAS, out)
    with pytest.raises(EvidenceError, match="new and outside"):
        analysis.write_analysis(exported, SCHEMAS, exported / "analysis")
    original = analysis._trace

    def mutate(*args):
        result = original(*args)
        (exported / "changed.txt").write_text("concurrent mutation", encoding="utf-8")
        return result

    monkeypatch.setattr(analysis, "_trace", mutate)
    with pytest.raises(EvidenceError, match="source changed"):
        analysis.write_analysis(exported, SCHEMAS, tmp_path / "changed")
    assert not (tmp_path / "changed").exists()
    with pytest.raises(EvidenceError, match="file set"):
        analysis.analyze(exported, SCHEMAS)


def test_intervention_sequences_repetition_and_unknown_settings_are_record_facts(
    tmp_path, monkeypatch
):
    events = []

    def add(event_type, **payload):
        events.append({"seq": len(events), "type": event_type, "payload": payload})

    add(
        "model.call",
        repaired_tool_calls=2,
        context_dropped_messages=3,
        context_truncated_tool_results=4,
    )
    add("supervisor.action", extensions={"stage": "benchmark_fault"})
    add("recovery")
    add("checkpoint.created")
    for _ in range(2):
        add("tool.call", capability_id="test.call", arguments={"x": 1})
    add("tool.result", status="error")
    add("environment.message", kind="preamble")
    add("environment.message", kind="continuation")
    monkeypatch.setattr(analysis, "_events", lambda _: events)
    record = {
        "event_sha256": "1" * 64,
        "model": {"settings": None},
        "final_state": "failed",
        "provider_terminal_codes": [],
    }
    trace = analysis._trace(tmp_path, {"initial_prompt": "fixture"}, record)
    assert trace["seqs"]["repairs"] == [0]
    assert trace["repaired_tool_calls"] == 2
    assert trace["context_dropped_messages"] == 3
    assert trace["context_truncated_tool_results"] == 4
    assert trace["seqs"]["marked_faults"] == [1]
    assert trace["seqs"]["recovery"] == [2]
    assert trace["seqs"]["checkpoints"] == [3]
    assert trace["identical_call_repeats"][0]["seqs"] == [4, 5]
    assert trace["model"]["settings"] is None
    assert not trace["provider_censored"]


def test_cli_and_wrong_pin_and_hostile_markdown_text(exported, tmp_path, capsys):
    out = tmp_path / "cli-analysis"
    assert (
        main(["evidence", "analyze", str(exported), "--schemas", str(SCHEMAS), "--out", str(out)])
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert (
        main(
            [
                "evidence",
                "verify-analysis",
                str(exported),
                str(out),
                "--sha256",
                result["analysis_sha256"],
            ]
        )
        == 0
    )
    assert main(["evidence", "verify-analysis", str(exported), str(out), "--sha256", "0" * 64]) == 2
    report = analysis.analyze(exported, SCHEMAS)
    report["groups"][0]["condition"] = "<script>![x](https://example.test)|\n"
    text = analysis.render(report).decode()
    assert "<script>" not in text
    assert "![x](" not in text


def test_annotation_input_change_keeps_stage_and_never_publishes(exported, tmp_path, monkeypatch):
    report = analysis.analyze(exported, SCHEMAS)
    path = tmp_path / "inputs/annotations.json"
    path.parent.mkdir()
    save(path, annotations(report))
    with pytest.raises(EvidenceError, match="annotation inputs"):
        analysis.write_analysis(exported, SCHEMAS, path.parent / "nested", path)
    original = shutil.copyfile
    changed = False

    def mutate(source, target):
        nonlocal changed
        result = original(source, target)
        if not changed:
            changed = True
            path.write_text("{}", encoding="utf-8")
        return result

    monkeypatch.setattr(analysis.shutil, "copyfile", mutate)
    with pytest.raises(EvidenceError, match="annotation input changed"):
        analysis.write_analysis(exported, SCHEMAS, tmp_path / "changed-annotation", path)
    assert not (tmp_path / "changed-annotation").exists()
    assert list(tmp_path.glob(".changed-annotation.staging-*"))


def test_all_unstarted_cells_stay_unknown_without_zero_filled_observations(study, root, tmp_path):
    path, spec = study
    spec["cells"] = [c for c in spec["cells"] if c["cell_id"] == "unstarted"]
    save(path, spec)
    directory = tmp_path / "unstarted-export"
    export_study(path, SCHEMAS, root, directory)
    report = analysis.analyze(directory, SCHEMAS)
    assert len(report["cells"]) == 1
    assert report["cells"][0]["trace"] is None
    assert report["groups"][0]["planned"] == 1
    assert report["groups"][0]["with_archive"] == 0
    assert report["contrasts"] == []
    assert report["validity_gates"][0]["mechanical_status"] == "findings"
    assert "unknown | unknown | unknown" in analysis.render(report).decode()
