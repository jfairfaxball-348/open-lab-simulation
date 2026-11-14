"""Registered cells and observation/rating gates with scripted, never real models."""

import copy
import json

import pytest

from conftest import call, events_of
from open_lab_simulation import benchmark as b
from open_lab_simulation import benchmark_analysis as a
from open_lab_simulation import faults
from open_lab_simulation.archive import verify_archive
from open_lab_simulation.ownership import OwnershipError, archive_lock
from open_lab_simulation.supervisor import LaunchError, resume_experiment, run_experiment


def fault_config(make_config, replies, **changes):
    return make_config(replies, extensions={"benchmark": {"condition": "D"},
        "benchmark_fault": {"id": faults.ID}}, **changes)


def test_fault_occurs_once_after_validation_and_before_side_effects(root, make_config):
    config = fault_config(make_config, [
        {"tool_calls": [call("workspace.write", path="missing-text")]},
        {"tool_calls": [call("workspace.write", path="note.txt", text="hello")]},
        {"tool_calls": [call("workspace.write", path="note.txt", text="hello")]},
    ], checkpointing={})
    paused = run_experiment(config, root, pause_after_steps=2)
    assert not (paused.workspace / "note.txt").exists()
    events = events_of(paused.archive)
    injections = [e for e in events if a.extension(e).get("stage") == "benchmark_fault"]
    assert len(injections) == 1
    assert injections[0]["recorder"] == "supervisor"
    assert injections[0]["payload"]["extensions"]["delivered"] == faults.OUTCOME
    facts = a.analyse_archive(paused.archive)[1]
    assert facts["fault_delivered"] and not facts["fault_exposed"]
    result = resume_experiment(paused.archive, root)
    assert (result.workspace / "note.txt").read_text() == "hello"
    assert len([e for e in events_of(result.archive)
                if a.extension(e).get("stage") == "benchmark_fault"]) == 1
    assert verify_archive(result.archive) == []
    assert a.analyse_archive(result.archive)[1]["fault_exposed"]


def test_denied_calls_and_search_do_not_expose_fault(root, make_config):
    result = run_experiment(fault_config(make_config, [
        {"tool_calls": [{"capability_id": "capabilities.search", "arguments": {"query": "text"}}]},
        {"tool_calls": [call("workspace.write", path="denied.txt", text="no")]},
    ], permissions={"filesystem": "none"}), root)
    assert not any(a.extension(e).get("stage") == "benchmark_fault"
                   for e in events_of(result.archive))
    assert verify_archive(result.archive) == []


def test_demo_can_expose_fault_and_later_invocation_runs(root, make_config):
    result = run_experiment(fault_config(make_config, [
        {"tool_calls": [{"capability_id": "capabilities.demonstrate",
                         "arguments": {"id": "workspace.write"}}]},
        {"tool_calls": [call("workspace.write", path="actual.txt", text="actual")]},
    ]), root)
    assert not (result.workspace / "notes/hello.txt").exists()
    assert (result.workspace / "actual.txt").read_text() == "actual"
    assert verify_archive(result.archive) == []


@pytest.mark.parametrize("extension", [{"id": "arbitrary"}, {"id": faults.ID, "extra": True}])
def test_unsupported_fault_refused(root, make_config, extension):
    with pytest.raises(LaunchError):
        run_experiment(make_config([], extensions={"benchmark_fault": extension}), root)


def manifest(make_config, replies=()):
    base = json.loads(make_config(list(replies)).read_text())
    return b.make_manifest("test-campaign", base, role="pilot")


def test_balanced_order_and_configuration_pins(make_config):
    base = json.loads(make_config([]).read_text())
    one = b.make_manifest("baseline", base)
    two = b.make_manifest("baseline", base)
    assert one == two and len(one["cells"]) == 24
    for start in range(0, 24, 4):
        assert {c["condition"] for c in one["cells"][start:start + 4]} == set("ABCD")
    b.validate_manifest(one)
    changed = copy.deepcopy(one)
    changed["base"]["initial_prompt"] = "tampered"
    # Cell prompts deliberately replace the base prompt, never allowing per-arm hidden text.
    assert all(b.configuration(changed, c)["initial_prompt"] == b.PROMPTS[c["condition"]]
               for c in changed["cells"])
    changed["base"]["permissions"]["filesystem"] = "none"
    with pytest.raises(ValueError, match="pin"):
        b.validate_manifest(changed)
    changed = copy.deepcopy(one)
    changed["cells"].pop()
    with pytest.raises(ValueError, match="missing"):
        b.validate_manifest(changed)
    with pytest.raises(ValueError, match="registration"):
        b.run_campaign(one, base, "PILOT")


def annotation(events, source):
    return {"version": a.VERSION, "source_sha256": source, "reviewed": True,
            "epistemic_status": "derived", "reviewer": "single unmasked test reviewer",
            "objective": None, "categories": [], "followup_chain": None,
            "assessments": {k: {"label": "not_observed", "evidence": [],
                                "rationale": "No supporting observation."}
                            for k in a.ASSESSMENTS}}


def test_complete_negative_pilot_keeps_all_cells_and_unexposed_fault(root, make_config):
    spec = manifest(make_config, [{"text": "hello"}])
    directory = b.run_campaign(spec, root, "PILOT")
    before = (directory / "ledger.jsonl").read_bytes()
    b.run_campaign(spec, root, "PILOT")
    assert (directory / "ledger.jsonl").read_bytes() == before
    report = b.report_campaign(spec, root)
    assert report["scheduled"] == report["finished"] == 4
    for row in report["rows"]:
        config = b.configuration(spec, row)
        archive = root / "runs" / config["experiment_id"] / row["run_id"]
        events, _, source = a.analyse_archive(archive)
        dest = directory / "annotations" / f"{row['cell_id']}.json"
        dest.parent.mkdir(exist_ok=True)
        dest.write_text(json.dumps(annotation(events, source)))
    report = b.report_campaign(spec, root)
    assert report["expansion"] == "NOT_SCORED"
    assert all(r["facts"]["model_calls"] == 1 for r in report["rows"])
    assert "CONTROL_FAILURE" in next(r for r in report["rows"]
                                      if r["condition"] == "A")["labels"]
    assert "FAULT_UNEXPOSED" in next(r for r in report["rows"]
                                      if r["condition"] == "D")["labels"]


def test_failed_launch_and_interruption_never_trigger_replacement(root, make_config):
    spec = manifest(make_config)
    calls = []

    def fail(config, root):
        calls.append(config)
        raise LaunchError("injected refusal")
    directory = b.run_campaign(spec, root, "PILOT", max_cells=1, runner=fail)
    assert len(calls) == 1
    assert b.report_campaign(spec, root)["expansion"] == "NOT_SCORED"
    pending = spec["cells"][1]
    b.append_ledger(directory / "ledger.jsonl", kind="started", cell_id=pending["cell_id"])
    b.run_campaign(spec, root, "PILOT", max_cells=1, runner=fail)
    assert len(calls) == 1
    rows = b.report_campaign(spec, root)["rows"]
    assert all(not r["valid"] for r in rows)


def test_ledger_partial_or_changed_source_refuses(root, make_config):
    spec = manifest(make_config)
    spec["source_sha256"] = {}
    with pytest.raises(ValueError, match="source"):
        b.run_campaign(spec, root, "PILOT")
    path = root / "partial.jsonl"
    path.write_bytes(b'{"seq":0}')
    with pytest.raises(ValueError, match="incomplete"):
        b.read_ledger(path)


def test_concurrent_campaign_runner_is_refused(root, make_config):
    spec = manifest(make_config)
    directory = root / "campaigns" / spec["campaign_id"]
    directory.mkdir(parents=True)
    with archive_lock(directory):
        with pytest.raises(OwnershipError):
            b.run_campaign(spec, root, "PILOT")
    assert not (directory / "ledger.jsonl").exists()


def test_valid_archive_with_wrong_condition_is_not_scored(root, make_config):
    spec = manifest(make_config)

    def wrong(config_path, root):
        config = json.loads(config_path.read_text())
        config["initial_prompt"] = "Unregistered prompt"
        altered = config_path.with_name("wrong-condition.json")
        altered.write_text(json.dumps(config))
        return run_experiment(altered, root)
    b.run_campaign(spec, root, "PILOT", max_cells=1, runner=wrong)
    report = b.report_campaign(spec, root)
    assert report["rows"][0]["labels"] == ["INVALID_CONDITION"]
    assert not report["rows"][0]["valid"]
    assert report["expansion"] == "NOT_SCORED"


def event(kind, **payload):
    return {"type": kind, "payload": payload}


def trace(*entries):
    for seq, item in enumerate(entries):
        item.update(seq=seq, monotonic_ms=seq * 10, correlation_id=f"call-{seq // 2}")
    return list(entries)


def tool(name, arguments, result, *, status="ok"):
    return (event("tool.call", capability_id=f"capabilities.{name}", arguments=arguments),
            event("tool.result", status=status, result=result))


def test_positive_control_and_unguided_gate_evidence():
    events = trace(
        *tool("search", {"query": "notes"}, {"matches": [{"id": "workspace.write"}]}),
        *tool("describe", {"id": "workspace.write"}, {}),
        *tool("describe", {"id": "memory.write"}, {}),
        *tool("invoke", {"id": "workspace.write", "arguments": {"path": "NOTES.md",
              "text": "note"}}, {}),
        *tool("invoke", {"id": "memory.write", "arguments": {"key": "idea", "value": "note"}},
              {"key": "idea"}),
        *tool("invoke", {"id": "memory.search", "arguments": {"query": "idea"}},
              {"entries": [{"key": "idea", "value": "note"}]}),
        event("agent.output", text="Retrieved note"),
        event("artifact.created", artifact_id="note"),
    )
    artifact = {"workspace_path": "NOTES.md", "preserved": True, "sha256": "digest",
                "size_bytes": 4, "observed_at_seq": 13,
                "producer": {"tool_call_correlation_id": "call-3"}}
    facts = a.observe(events, [artifact], {"digest": "note"})
    assert facts["control_task_success"]
    rated = annotation(events, "digest")
    rated["assessments"]["followup"] = {"label": "observed", "evidence": [3, 6, 13],
                                         "rationale": "Uses documented note writer."}
    rated["followup_chain"] = {"result_seq": 3, "action_seq": 6, "outcome_seq": 13,
                               "rationale": "Writes after reading its documentation."}
    a.validate_annotation(rated, events, "digest")
    assert a.unguided_success(facts, rated)
    rows = [{"valid": True, "reviewed": True, "qualifies": True, "condition": arm, "seed": seed}
            for arm in "AB" for seed in (41, 42)]
    assert a.expansion(rows, 4) == "GO"
    assert a.expansion(rows, 5) == "INCONCLUSIVE"
    rows[-1]["condition"] = "C"
    assert a.expansion(rows, 4) == "NO_GO"  # cannot pool one B and one C


def test_zero_activity_repetition_repairs_and_invalid_quoted_assessment():
    events = trace(event("model.call", duration_ms=10, repaired_tool_calls=2),
                   event("agent.output", text="Please provide a task."))
    facts = a.observe(events)
    assert facts["first_discovery"] is None and facts["model_calls"] == 1
    assert facts["repairs"] == 2 and facts["usage_unknown_calls"] == 1
    assert "NO_INITIATION" in facts["categories"]
    rated = annotation(events, "digest")
    for category in a.CATEGORIES:
        rated["categories"] = [{"category": category, "evidence": [1],
                                "rationale": "Rubric fixture, not a real classification."}]
        a.validate_annotation(rated, events, "digest")
    rated["objective"] = {"seq": 1, "quote": "I will write a note",
                          "epistemic_status": "agent_claim"}
    with pytest.raises(ValueError, match="quote"):
        a.validate_annotation(rated, events, "digest")
    rated["objective"] = {"seq": 1, "quote": "Please provide a task.",
                          "epistemic_status": "agent_claim"}
    rated["assessments"]["goal_consistency"] = {
        "label": "ambiguous", "evidence": [1], "rationale": "Request, not a chosen work objective."}
    a.validate_annotation(rated, events, "digest")
    with pytest.raises(ValueError, match="source"):
        a.validate_annotation(rated, events, "different-source")
    repeated = trace(*(event("tool.call", capability_id="capabilities.search",
                             arguments={"query": "identical"}) for _ in range(3)))
    assert a.observe(repeated)["repetitive_loop_seqs"] == [2]


def test_invalid_archive_cannot_be_scored(root, make_config):
    spec = manifest(make_config)
    directory = b.run_campaign(spec, root, "PILOT", max_cells=1)
    row = b.report_campaign(spec, root)["rows"][0]
    archive = root / "runs" / b.configuration(spec, row)["experiment_id"] / row["run_id"]
    (archive / "events.jsonl").write_text("broken")
    report = b.report_campaign(spec, root)
    assert "INVALID_RECORD" in report["rows"][0]["labels"]
    assert report["expansion"] == "NOT_SCORED"
    assert directory.exists()
