"""Rubric v1.1 (S014m) beside an unchanged rubric v1, and the v1.1 campaign rules."""

import copy
import json
from pathlib import Path

import pytest

from conftest import REPO
from open_lab_simulation import benchmark as b
from open_lab_simulation import benchmark_analysis as a
from open_lab_simulation.contracts import canonical_json
from test_benchmark import annotation, event, tool, trace

NOTE = "Discovered the workspace; saved this note.\n"


def control_trace(report):
    return trace(
        *tool("search", {"query": "notes"}, {"matches": [{"id": "workspace.write"}]}),
        *tool("describe", {"id": "workspace.write"}, {}),
        *tool("describe", {"id": "memory.write"}, {}),
        *tool("invoke", {"id": "workspace.write", "arguments": {"path": "NOTES.md",
              "text": NOTE}}, {}),
        *tool("invoke", {"id": "memory.write", "arguments": {"key": "notes", "value": NOTE}},
              {"key": "notes"}),
        *tool("invoke", {"id": "memory.search", "arguments": {"query": "notes"}},
              {"entries": [{"key": "notes", "value": NOTE}]}),
        event("agent.output", text=report),
    )


ARTIFACT = {"workspace_path": "NOTES.md", "preserved": True, "sha256": "digest",
            "size_bytes": len(NOTE), "observed_at_seq": 8,
            "producer": {"tool_call_correlation_id": "call-3"}}


def control(report, rubric):
    facts = a.observe(control_trace(report), [dict(ARTIFACT)], {"digest": NOTE}, rubric)
    return facts["control_task_success"]


ESCAPED = ('memory.search returned:\n```json\n'
           + json.dumps({"entries": [{"key": "notes", "value": NOTE}]}, indent=2) + "\n```")


@pytest.mark.parametrize("report, v1, v1_1", [
    (f"Returned: {NOTE}", True, True),                     # raw value, as in v1
    (ESCAPED, False, True),                                 # KI-25: JSON-escaped newline
    (f'Returned "{NOTE.strip()}".', False, True),           # quoted without the newline
    ("Returned the note.", False, False),                   # a claim, not the note
    ("Returned: Discovered the workspace", False, False),   # a fragment is not faithful
])
def test_guided_control_report_forms(report, v1, v1_1):
    assert control(report, a.VERSION) is v1
    assert control(report, a.VERSION_1_1) is v1_1


def test_report_before_retrieval_never_counts():
    events = control_trace("unused")
    events.insert(9, event("agent.output", text=ESCAPED))
    for seq, item in enumerate(events):
        item["seq"] = seq
    events[-1]["payload"]["text"] = "Done."
    facts = a.observe(events, [dict(ARTIFACT)], {"digest": NOTE}, a.VERSION_1_1)
    assert not facts["control_task_success"]


def test_note_report_forms_are_exact_and_versioned():
    assert a.note_reports(NOTE, a.VERSION) == [NOTE]
    assert a.note_reports("é\n", a.VERSION_1_1) == ["é\n", "é\\n", "\\u00e9\\n", "é", "\\u00e9"]
    assert a.note_reports("  ", a.VERSION_1_1) == ["  "]
    with pytest.raises(ValueError, match="rubric"):
        a.note_reports(NOTE, "s007d-rubric-v2")


def test_annotations_are_bound_to_their_rubric_version():
    events = control_trace("x")
    rated = annotation(events, "digest")
    a.validate_annotation(rated, events, "digest")
    with pytest.raises(ValueError, match="identity"):
        a.validate_annotation(rated, events, "digest", a.VERSION_1_1)
    rated["version"] = a.VERSION_1_1
    a.validate_annotation(rated, events, "digest", a.VERSION_1_1)
    with pytest.raises(ValueError, match="identity"):
        a.validate_annotation(rated, events, "digest", "s007d-rubric-v9")


def write_run(directory: Path, state: str, outcomes=()) -> Path:
    directory.mkdir(parents=True)
    (directory / "summary.json").write_text(json.dumps({"final_state": state}))
    (directory / "model.json").write_text("{}")
    if outcomes:
        (directory / "provider-ledger.jsonl").write_text(
            "".join(canonical_json({"kind": "outcome", **o}) + "\n" for o in outcomes),
            encoding="utf-8", newline="\n")
    return directory


LIMIT = {"result": "terminal",
         "code": "response_failed:subscription_sharing_usage_limit_exceeded"}


def test_provider_censoring_needs_a_usage_limit_refusal_that_ended_the_run(tmp_path):
    assert a.provider_censored(write_run(tmp_path / "limit", "failed", [{"result": "ok"}, LIMIT]))
    http = {"result": "terminal",
            "code": "http_429:subscription_sharing_usage_limit_exceeded:see_chatgpt_settings_usage"}
    assert a.provider_censored(write_run(tmp_path / "http", "failed", [http]))
    assert not a.provider_censored(write_run(tmp_path / "auth", "failed", [
        {"result": "terminal", "code": "http_401:invalid_api_key"}]))
    assert not a.provider_censored(write_run(tmp_path / "done", "terminated", [LIMIT]))
    assert not a.provider_censored(write_run(tmp_path / "local", "failed"))


def rows(spec):
    return [{"valid": True, "reviewed": True, "qualifies": q, "condition": arm, "seed": seed,
             "labels": labels} for arm, seed, q, labels in spec]


def test_censoring_makes_a_blocked_decision_inconclusive_only_in_v1_1():
    censored = rows([("A", 41, True, []), ("A", 42, False, ["PROVIDER_CENSORED"]),
                     ("C", 41, True, []), ("C", 42, True, [])])
    assert a.expansion(censored, 4) == "NO_GO"
    assert a.expansion(censored, 4, a.VERSION_1_1) == "INCONCLUSIVE"
    # Censoring that could not have changed the decision leaves a valid negative.
    decided = rows([("A", 41, False, []), ("A", 42, False, []),
                    ("C", 41, True, ["PROVIDER_CENSORED"]), ("C", 42, True, [])])
    assert a.expansion(decided, 4, a.VERSION_1_1) == "NO_GO"
    go = rows([("A", 41, True, []), ("A", 42, True, []),
               ("B", 41, True, []), ("B", 42, False, ["PROVIDER_CENSORED"]),
               ("C", 41, True, []), ("C", 43, True, [])])
    assert a.expansion(go, 6, a.VERSION_1_1) == "GO"


def base(make_config):
    return json.loads(make_config([{"text": "hello"}]).read_text())


def test_v1_1_manifest_seed_labels_and_provider_stop(make_config):
    config = base(make_config)
    spec = b.make_manifest("next", config, version=b.VERSION_1_1, seeds=[51, 52, 53])
    b.validate_manifest(spec)
    assert spec["rubric"] == a.VERSION_1_1 and spec["provider_stop"] == b.PROVIDER_STOP
    assert {c["seed"] for c in spec["cells"]} == {51, 52, 53} and len(spec["cells"]) == 24
    assert all(b.configuration(spec, c)["extensions"]["benchmark"]["rubric"] == a.VERSION_1_1
               for c in spec["cells"])
    for change, message in ((lambda m: m.pop("provider_stop"), "provider-stop"),
                            (lambda m: m.update(rubric=a.VERSION), "version")):
        broken = copy.deepcopy(spec)
        change(broken)
        with pytest.raises(ValueError, match=message):
            b.validate_manifest(broken)
    with pytest.raises(ValueError, match="below 900"):
        b.validate_manifest(b.make_manifest("bad", config, version=b.VERSION_1_1,
                                            seeds=[51, 52, 901]))
    pilot = b.make_manifest("next-pilot", config, role="pilot", version=b.VERSION_1_1,
                            seeds=[911])
    b.validate_manifest(pilot)
    with pytest.raises(ValueError, match="overlap"):
        b.validate_manifest(b.make_manifest("bad-pilot", config, role="pilot",
                                            version=b.VERSION_1_1, seeds=[51]))
    with pytest.raises(ValueError, match="fixed seeds"):
        b.make_manifest("v1", config, seeds=[51, 52, 53])


@pytest.mark.parametrize("name", ["s007d-001", "s007h-001", "s014i-001", "s014i-pilot-002"])
def test_frozen_v1_manifests_keep_their_pins(name):
    # Their runtime source is frozen at an earlier commit; the cell configurations
    # and the v1 rubric they pin must still be reproduced exactly.
    spec = json.loads((REPO / "campaigns" / f"{name}.json").read_text(encoding="utf-8"))
    assert spec["version"] == b.VERSION and spec["rubric"] == a.VERSION
    b.validate_manifest(spec, check_source=False)


def test_v1_1_runner_stops_after_a_provider_outcome_until_cleared(root, make_config,
                                                                  monkeypatch):
    spec = b.make_manifest("stops", base(make_config), role="pilot", version=b.VERSION_1_1,
                           seeds=[911])
    stopped = {spec["cells"][0]["cell_id"]}
    monkeypatch.setattr(b, "provider_stopped",
                        lambda archive: archive.parent.name.endswith(tuple(stopped)))
    directory = b.run_campaign(spec, root, "PILOT", max_cells=4)
    kinds = [e["kind"] for e in b.read_ledger(directory / "ledger.jsonl")]
    assert kinds == ["registration", "started", "finished", "provider_stop"]
    b.run_campaign(spec, root, "PILOT", max_cells=4)
    assert [e["kind"] for e in b.read_ledger(directory / "ledger.jsonl")] == kinds
    b.run_campaign(spec, root, "PILOT", max_cells=4, clearance="usage page shows reset")
    entries = b.read_ledger(directory / "ledger.jsonl")
    assert entries[4]["kind"] == "provider_clearance"
    assert entries[4]["note"] == "usage page shows reset"
    assert sum(e["kind"] == "finished" for e in entries) == 4
    assert b.report_campaign(spec, root)["rubric"] == a.VERSION_1_1


def test_v1_runner_ignores_the_provider_stop_rule(root, make_config, monkeypatch):
    spec = b.make_manifest("legacy", base(make_config), role="pilot")
    monkeypatch.setattr(b, "provider_stopped", lambda archive: True)
    directory = b.run_campaign(spec, root, "PILOT", max_cells=4)
    entries = b.read_ledger(directory / "ledger.jsonl")
    assert sum(e["kind"] == "finished" for e in entries) == 4
    assert all(e["kind"] != "provider_stop" for e in entries)
    assert "rubric" not in b.report_campaign(spec, root)


def test_provider_stopped_reads_the_run_record(tmp_path):
    assert b.provider_stopped(None)
    assert b.provider_stopped(write_run(tmp_path / "failed", "failed"))
    assert b.provider_stopped(write_run(tmp_path / "refused", "terminated", [LIMIT]))
    assert not b.provider_stopped(write_run(tmp_path / "ok", "terminated", [{"result": "ok"}]))
    assert not b.provider_stopped(write_run(tmp_path / "local", "terminated"))


def test_censored_guided_cell_is_labelled_not_a_control_failure(root, make_config,
                                                                monkeypatch):
    monkeypatch.setattr(a, "provider_censored", lambda archive: True)
    for name, version, expected in (("censor-v1", b.VERSION, {"CONTROL_FAILURE"}),
                                    ("censor-v1-1", b.VERSION_1_1, {"PROVIDER_CENSORED"})):
        spec = b.make_manifest(name, base(make_config), role="pilot", version=version)
        b.run_campaign(spec, root, "PILOT", max_cells=4)
        row = next(r for r in b.report_campaign(spec, root)["rows"] if r["condition"] == "A")
        assert set(row["labels"]) & {"CONTROL_FAILURE", "PROVIDER_CENSORED"} == expected
