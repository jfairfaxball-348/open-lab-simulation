"""Bounded active-time scheduling and operator-only data lineage."""
import copy
import json
import shutil
import sys
import time
from pathlib import Path

import pytest

from conftest import call, events_of
from open_lab_simulation import policies, supervisor
from open_lab_simulation.archive import verify_archive
from open_lab_simulation.artifacts import read_index
from open_lab_simulation.checkpoints import checkpoint_events, read_checkpoint, run_lock
from open_lab_simulation.run_time import Clock, verify
from open_lab_simulation.supervisor import LaunchError, resume_experiment, run_experiment
from open_lab_simulation.world import load_world
from test_container import backend  # noqa: F401

REPO = Path(__file__).resolve().parent.parent
RELEASE = REPO / "tests/fixtures/worlds/habitat/0.5.0"
PARENT = REPO / "experiments/habitat/s018-parent-demo.json"
CHILD = REPO / "experiments/habitat/s018-child-demo.json"
CONFIG = json.loads(PARENT.read_text(encoding="utf-8"))


@pytest.fixture
def time_root(tmp_path):
    root = tmp_path / "data"
    shutil.copytree(RELEASE, root / "worlds/habitat/0.5.0")
    return root


def config_file(tmp_path, replies, *, name="time-check", **changes):
    value = copy.deepcopy(CONFIG)
    value["experiment_id"] = name
    value.update(changes)
    value.setdefault("extensions", {}).setdefault("scripted_model", {})["replies"] = replies
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def outcomes(archive):
    return [e["payload"] for e in events_of(archive) if e["type"] == "tool.result"]


def checkpoint(archive):
    events = events_of(archive)
    return read_checkpoint(archive, checkpoint_events(events)[-1], events,
                           load_world(RELEASE).contracts)[1]


class Records:
    def __init__(self, fail=False):
        self.events, self.fail = [], fail

    def record(self, **event):
        self.events.append(event)

    def sync(self):
        if self.fail:
            raise OSError("record failure")


def clock_fixture():
    elapsed = [3.0]
    clock = Clock(CONFIG, load_world(RELEASE), policies.resolve(CONFIG, "continue"),
                  lambda: elapsed[0])
    return clock, elapsed, Records()


def test_exact_budget_wake_boundary_and_ready_checkpoint_latch():
    clock, elapsed, records = clock_fixture()
    result = clock.execute("time.remaining", {}, records, {"kind": "supervisor"}, "call-1")
    assert result["result"] == {"available": True, "elapsed_ms": 3000,
                                "remaining_ms": 597000, "budget_ms": 600000}
    asleep = clock.execute("time.sleep", {"seconds": 2}, records,
                           {"kind": "supervisor"}, "call-2")
    assert asleep["result"]["wake_at_elapsed_ms"] == 5000
    elapsed[0] = 4.999
    clock.barrier({"idle": True}, records, {"kind": "supervisor"})
    assert clock.pending()
    elapsed[0] = 5
    clock.barrier({"idle": True}, records, {"kind": "supervisor"})
    assert not clock.pending() and clock.state["wake_ready"]
    restored = Clock(CONFIG, clock.world, clock.policy, lambda: elapsed[0], clock.snapshot())
    restored.barrier({"idle": True}, records, {"kind": "supervisor"})
    assert restored.state["wake_ready"]
    restored.barrier({"idle": False}, records, {"kind": "supervisor"})
    assert restored.snapshot() == {"wake_at_elapsed_ms": None, "wake_ready": False}
    assert sum(e["payload"]["extensions"]["stage"] == "time_wake" for e in records.events) == 1


@pytest.mark.parametrize("arguments", [{"seconds": 0}, {"seconds": -1}, {"seconds": True},
                                        {"seconds": 3601}, {"seconds": 1, "budget": 99999}])
def test_invalid_sleep_does_not_change_authority(arguments):
    clock, _, records = clock_fixture()
    before = clock.snapshot()
    assert clock.execute("time.sleep", arguments, records, {}, "call-1")["status"] == "error"
    assert clock.snapshot() == before and not records.events


def test_policy_budget_duplicate_and_record_failure_refuse_without_effects():
    clock, elapsed, records = clock_fixture()
    assert clock.execute("time.sleep", {"seconds": 6}, records, {}, "call-1")["status"] == "error"
    elapsed[0] = 599
    assert clock.execute("time.sleep", {"seconds": 1}, records, {}, "call-2")["status"] == "error"
    elapsed[0] = 3
    with pytest.raises(OSError):
        clock.execute("time.sleep", {"seconds": 1}, Records(fail=True), {}, "call-3")
    assert not clock.pending()
    assert clock.execute("time.sleep", {"seconds": 1}, records, {}, "call-4")["status"] == "ok"
    target = clock.snapshot()
    assert clock.execute("time.sleep", {"seconds": 1}, records, {}, "call-5")["status"] == "error"
    assert clock.snapshot() == target
    legacy = {**clock.policy, "id": policies.LEGACY}
    other = Clock(CONFIG, clock.world, legacy, lambda: 0)
    assert other.execute("time.sleep", {"seconds": 1}, records, {}, "call-6")["status"] == "denied"


@pytest.mark.parametrize("state", [
    {"wake_at_elapsed_ms": 600000, "wake_ready": False},
    {"wake_at_elapsed_ms": True, "wake_ready": False},
    {"wake_at_elapsed_ms": 1000, "wake_ready": True}, {"forged": True},
])
def test_private_wake_state_cannot_extend_budget(state):
    with pytest.raises(ValueError):
        Clock(CONFIG, load_world(RELEASE), policies.resolve(CONFIG, "continue"), lambda: 0, state)


def test_sleep_pause_resume_has_no_inference_or_messages_during_wait(time_root, backend):  # noqa: F811
    started = time.perf_counter()
    paused = run_experiment(PARENT, time_root, pause_after_steps=6)
    assert paused.state == "paused", paused.reason
    private = checkpoint(paused.archive)["time_state"]
    assert private["wake_at_elapsed_ms"] is not None
    before = (paused.archive / "events.jsonl").read_bytes()
    time.sleep(2.2)  # longer than requested sleep, but paused downtime is excluded
    resumed = resume_experiment(paused.archive, time_root)
    assert resumed.state == "completed", resumed.reason
    elapsed = json.loads((resumed.archive / "summary.json").read_text())["elapsed_s"]
    assert time.perf_counter() - started - elapsed >= 2.0
    assert (resumed.archive / "events.jsonl").read_bytes().startswith(before)
    found = outcomes(resumed.archive)
    assert all(r["status"] == "ok" for r in found), found
    assert found[6]["result"]["elapsed_ms"] >= found[5]["result"]["wake_at_elapsed_ms"]
    assert found[6]["result"]["remaining_ms"] < found[4]["result"]["remaining_ms"]
    assert (resumed.workspace / "result.txt").read_text() == "10"
    events = events_of(resumed.archive)
    stages = [(e, e["payload"].get("extensions", {}).get("stage")) for e in events]
    scheduled = next(e for e, stage in stages if stage == "time_sleep_scheduled")
    wake = next(e for e, stage in stages if stage == "time_wake")
    during = [e for e in events if scheduled["seq"] < e["seq"] < wake["seq"]]
    assert not any(e["type"] in ("model.call", "agent.output", "environment.message")
                   for e in during)
    assert sum(e["type"] == "environment.message" for e in events) == 1  # original preamble only
    assert checkpoint(resumed.archive)["time_state"] == {
        "wake_at_elapsed_ms": None, "wake_ready": False}
    assert verify_archive(resumed.archive) == []
    altered = copy.deepcopy(events)
    wake_event = next(e for e in altered if e["payload"].get("extensions", {}).get(
        "stage") == "time_wake")
    wake_event["payload"]["extensions"]["elapsed_ms"] = 0
    assert verify(CONFIG, policies.resolve(CONFIG, "continue"), altered)[0]
    assert backend.running(resumed.run_id) == []


def test_past_budget_sleep_is_refused_and_deadline_still_wins(time_root, tmp_path, backend):  # noqa: F811
    path = config_file(tmp_path, [
        {"tool_calls": [call("time.sleep", seconds=5)]},
        {"text": "No more activity."}], limits={"wall_clock_s": 2, "max_steps": 10})
    result = run_experiment(path, time_root)
    assert result.state == "terminated" and result.reason == "deadline"
    assert outcomes(result.archive)[0]["status"] == "error"
    assert not any(e["payload"].get("extensions", {}).get("stage") == "time_wake"
                   for e in events_of(result.archive))
    assert verify_archive(result.archive) == []


def test_same_turn_calls_after_sleep_are_denied(time_root, tmp_path, backend):  # noqa: F811
    path = config_file(tmp_path, [
        {"tool_calls": [call("time.sleep", seconds=1),
                        call("workspace.write", path="should-not-exist", text="bad")]},
        {"tool_calls": [call("time.remaining")]},
    ])
    result = run_experiment(path, time_root)
    assert result.state == "completed", result.reason
    assert [r["status"] for r in outcomes(result.archive)] == ["ok", "denied", "ok"]
    assert not (result.workspace / "should-not-exist").exists()
    assert verify_archive(result.archive) == []


def test_lineage_imports_final_data_not_live_files_or_authority(time_root, tmp_path, backend):  # noqa: F811
    schema = {"type": "object", "properties": {"value": {"type": "integer"}},
              "required": ["value"], "additionalProperties": False}
    definition = {"id": "agent.double", "source_path": "project.py", "name": "Double",
                  "summary": "Double an integer.", "description": "Double the integer argument.",
                  "input_schema": schema, "output_schema": schema,
                  "examples": [{"input": {"value": 2}, "expected_output": {"value": 4}}]}
    parent_path = config_file(tmp_path, [
        {"tool_calls": [call("workspace.write", path="project.py",
                             text='def run(a): return {"value": a["value"] * 2}\n')]},
        {"tool_calls": [call("tools.create", **definition)]},
        {"tool_calls": [call("memory.write", key="project", value="Remember project.py")]},
        {"tool_calls": [call("world.act", x=0, y=0, drive=3)]},
        {"tool_calls": [call("world.advance", ticks=2)]},
    ], capabilities={"agent_created_tools": True},
        extensions={**CONFIG["extensions"], "phenomena": {"seed": 41}})
    parent = run_experiment(parent_path, time_root)
    assert parent.state == "completed", parent.reason
    log, summary = (parent.archive / "events.jsonl").read_bytes(), (
        parent.archive / "summary.json").read_bytes()
    (parent.workspace / "project.py").write_text("LIVE EDIT MUST NOT BE INHERITED",
                                                encoding="utf-8")
    child_path = config_file(tmp_path, [
        {"tool_calls": [call("workspace.read", path="project.py")]},
        {"tool_calls": [call("memory.search", query="Remember")]},
        {"tool_calls": [call("world.observe")]},
        {"tool_calls": [call("time.remaining")]},
        {"tool_calls": [{"capability_id": "capabilities.describe",
                         "arguments": {"id": "agent.double"}}]},
        {"tool_calls": [call("workspace.write", path="denied", text="bad")]},
    ], name="child-check", limits={"wall_clock_s": 300, "max_steps": 10},
        permissions={"filesystem": "workspace_read", "network": "none", "subprocess": False})
    child_config = json.loads(child_path.read_text(encoding="utf-8"))
    child_config["extensions"]["phenomena"] = {"seed": 41}
    child_path.write_text(json.dumps(child_config), encoding="utf-8")
    child = run_experiment(child_path, time_root, from_run=parent.archive)
    assert child.state == "completed", child.reason
    found = outcomes(child.archive)
    assert 'def run(a)' in found[0]["result"]["text"]
    assert found[1]["result"]["entries"][0]["value"] == "Remember project.py"
    assert found[2]["result"]["tick"] == 0
    assert all(c["drive"] == 0 for c in found[2]["result"]["cells"])
    assert found[3]["result"]["budget_ms"] == 300000
    assert found[4]["status"] == "error" and found[5]["status"] == "denied"
    assert "agent_tools" not in checkpoint(child.archive)
    first_model = next(e["seq"] for e in events_of(child.archive) if e["type"] == "model.call")
    inherited = next(item for item in read_index(child.archive)
                     if item["workspace_path"] == "project.py")
    assert inherited["producer"] == {"kind": "system"}
    assert inherited["observed_at_seq"] < first_model
    assert (parent.archive / "events.jsonl").read_bytes() == log
    assert (parent.archive / "summary.json").read_bytes() == summary
    assert (parent.archive / "children" / f"{child.run_id}.json").is_file()
    assert verify_archive(parent.archive) == verify_archive(child.archive) == []
    portable = tmp_path / "portable" / child.run_id
    shutil.copytree(child.archive, portable)
    assert verify_archive(portable) == []
    source_zip = next(portable.glob("lineage-source/checkpoints/*/workspace.zip"))
    source_zip.write_bytes(b"tampered")
    assert any("lineage" in issue for issue in verify_archive(portable))
    with run_lock(parent.archive):
        with pytest.raises(LaunchError, match="lineage import refused"):
            run_experiment(child_path, time_root, from_run=parent.archive)


def test_lineage_refuses_paused_corrupt_and_outside_sources(time_root, tmp_path, backend):  # noqa: F811
    path = config_file(tmp_path, [{"tool_calls": [call("time.remaining")]}])
    paused = run_experiment(path, time_root, pause_after_steps=1)
    assert paused.state == "paused"
    with pytest.raises(LaunchError, match="completed"):
        run_experiment(CHILD, time_root, from_run=paused.archive)
    with pytest.raises(LaunchError, match="data root"):
        run_experiment(CHILD, time_root, from_run=tmp_path / "outside")
    completed = resume_experiment(paused.archive, time_root)
    assert completed.state == "completed"
    source_zip = next(completed.archive.glob("checkpoints/*/workspace.zip"))
    source_zip.write_bytes(b"tampered")
    with pytest.raises(LaunchError, match="verification"):
        run_experiment(CHILD, time_root, from_run=completed.archive)
    no_snapshot_path = config_file(tmp_path, [{"tool_calls": [call("time.remaining")]}],
                                   name="no-snapshot")
    document = json.loads(no_snapshot_path.read_text(encoding="utf-8"))
    document.pop("checkpointing")
    no_snapshot_path.write_text(json.dumps(document), encoding="utf-8")
    no_snapshot = run_experiment(no_snapshot_path, time_root)
    assert no_snapshot.state == "completed", no_snapshot.reason
    with pytest.raises(LaunchError, match="checkpoint"):
        run_experiment(CHILD, time_root, from_run=no_snapshot.archive)


def test_legacy_and_quiet_cannot_enable_scheduled_sleep():
    for ident in (policies.LEGACY, policies.QUIET):
        value = copy.deepcopy(CONFIG)
        value["extensions"]["runtime_policy"] = {"id": ident, "max_sleep_s": 5}
        with pytest.raises(ValueError, match="self-scheduled"):
            policies.resolve(value, "continue")


def test_worker_cannot_forge_private_wake_or_start_inference_early(
        time_root, tmp_path, backend, monkeypatch):  # noqa: F811
    script = tmp_path / "hostile_clock_worker.py"
    script.write_text('''import json, sys, time
from open_lab_simulation.policies import digest
spec = json.loads(sys.stdin.readline())
assert 'time_state' not in spec.get('runtime_state', {})
def send(message):
    print(json.dumps(message), flush=True)
    return json.loads(sys.stdin.readline())
args = {'seconds': 1}
send({'event': {'recorder':'tool_host','type':'tool.call','actor':{'kind':'agent'},
               'correlation_id':'call-000001',
               'payload':{'capability_id':'time.sleep','arguments':args}}})
wrong = send({'time_service': {'id':'time.sleep','arguments':{'seconds':2}}})
assert wrong['time_result']['status'] == 'denied'
outcome = send({'time_service': {'id':'time.sleep','arguments':args}})['time_result']
assert outcome['status'] == 'ok'
again = send({'time_service': {'id':'time.sleep','arguments':args}})['time_result']
assert again['status'] == 'denied'
send({'event': {'recorder':'tool_host','type':'tool.result','actor':{'kind':'system'},
               'correlation_id':'call-000001','payload':outcome}})
assert send({'model_input': {'request':{'messages':[]},'context':{}}})['seq'] is None
state = {'messages':[{'role':'system','content':'test'},{'role':'user','content':'test'}],
         'steps':1,'script_cursor':0,'call_count':1,'tokens_used':0,
         'policy_sha256':digest(spec['model']['runtime_policy']),
         'policy_state':{'idle':True,'tool_free_replies':0},
         'time_state':{'wake_at_elapsed_ms':999999999,'wake_ready':True}}
while not send({'state':state})['wake']:
    time.sleep(.25)
state['policy_state']['idle'] = False
send({'state':state,'final':True})
print(json.dumps({'end':{'reason':'model_finished','steps':1}}),flush=True)
''', encoding="utf-8")
    monkeypatch.setattr(supervisor, "WORKER_COMMAND", [sys.executable, str(script)])
    result = run_experiment(PARENT, time_root)
    assert result.state == "completed", result.notes
    assert checkpoint(result.archive)["time_state"] == {
        "wake_at_elapsed_ms": None, "wake_ready": False}
    assert verify_archive(result.archive) == []
