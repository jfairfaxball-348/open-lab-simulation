"""Deterministic laws, supervisor-only authority, private checkpointing and replay."""
import copy
import json
import sys
from pathlib import Path

import pytest

from conftest import call, events_of
from open_lab_simulation import phenomena, supervisor
from open_lab_simulation.archive import verify_archive
from open_lab_simulation.checkpoints import checkpoint_events, read_checkpoint
from open_lab_simulation.phenomena import Field, Service, digest, policy, replay
from open_lab_simulation.supervisor import LaunchError, resume_experiment, run_experiment
from open_lab_simulation.world import load_world
from test_container import backend  # noqa: F401

REPO = Path(__file__).resolve().parent.parent
RELEASE = REPO / "tests/fixtures/worlds/habitat/0.4.0"
DEMO = REPO / "experiments/habitat/s017-demo.json"
CONFIG = json.loads(DEMO.read_text(encoding="utf-8"))


@pytest.fixture
def field_root(tmp_path):
    import shutil

    root = tmp_path / "data"
    shutil.copytree(RELEASE, root / "worlds/habitat/0.4.0")
    return root


def configured(tmp_path, replies, **changes):
    config = copy.deepcopy(CONFIG)
    config["extensions"]["scripted_model"]["replies"] = replies
    config.update(changes)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def results(archive):
    return [e["payload"] for e in events_of(archive) if e["type"] == "tool.result"]


def checkpoint_state(archive):
    events = events_of(archive)
    event = checkpoint_events(events)[-1]
    return read_checkpoint(archive, event, events, load_world(RELEASE).contracts)[1]


def test_hand_computed_synchronous_transport_and_torus():
    field = Field(0)
    field.state.update(water=[320] * 36, activity=[24] * 36)
    original = field.observe()
    field.apply("world.act", {"x": 0, "y": 0, "drive": 3})
    assert field.observe()["tick"] == 0
    assert field.observe()["cells"][0]["water"] == original["cells"][0]["water"]
    first = field.apply("world.advance", {"ticks": 1})
    assert first["cells"][0]["water"] == 341
    assert {c["water"] for c in first["cells"][1:]} == {317}
    assert {c["activity"] for c in first["cells"]} == {24}
    second = field.apply("world.advance", {"ticks": 1})
    assert second["cells"][0]["water"] == 356
    assert {second["cells"][i]["water"] for i in (1, 5, 6, 30)} == {315}
    assert second["cells"][7]["water"] == 314


def test_seeds_noop_reads_copy_isolation_and_bounded_values():
    first, other = Field(41), Field(42)
    assert first.state == Field(41).state and first.state != other.state
    # A versioned initial-state vector catches accidental seed/serialization drift.
    assert digest(first.state) == "d10257320b2a29d86f366fafed5e8614ec6553c2efc700d04fb440d4b9657cca"
    assert first.observe()["cells"][0] == {"x": 0, "y": 0, "water": 308,
                                           "activity": 63, "drive": 0}
    before = copy.deepcopy(first.state)
    observation = first.apply("world.observe", {})
    observation["cells"][0]["water"] = -999
    assert first.state == before
    for x in range(6):
        first.apply("world.act", {"x": x, "y": 0, "drive": -3})
        first.apply("world.act", {"x": x, "y": 5, "drive": 3})
    for _ in range(30):
        first.apply("world.advance", {"ticks": 100})
    assert first.state["tick"] == 3000
    assert min(first.state["water"]) >= 0 and max(first.state["water"]) <= 1000
    assert min(first.state["activity"]) >= 0 and max(first.state["activity"]) <= 200
    restored = Field(41, first.state)
    first.apply("world.advance", {"ticks": 10})
    assert restored.apply("world.advance", {"ticks": 10}) == first.observe()


@pytest.mark.parametrize("ident, arguments", [
    ("world.observe", {"seed": 2}), ("world.act", {"x": 6, "y": 0, "drive": 0}),
    ("world.act", {"x": 0, "y": 0, "drive": 4}),
    ("world.act", {"x": True, "y": 0, "drive": 0}),
    ("world.advance", {"ticks": 0}), ("world.advance", {"ticks": 101}),
    ("world.advance", {"ticks": "exec(...)"}), ("world.reset", {}),
])
def test_invalid_operations_have_no_effect(ident, arguments):
    field = Field(41)
    before = copy.deepcopy(field.state)
    with pytest.raises(ValueError):
        field.apply(ident, arguments)
    assert field.state == before


def test_total_tick_bound_and_private_state_refusal():
    field = Field(41)
    field.state["tick"] = phenomena.MAX_TICK - 1
    before = copy.deepcopy(field.state)
    with pytest.raises(ValueError, match="bounds"):
        field.apply("world.advance", {"ticks": 2})
    assert field.state == before
    for changed in ({"seed": 42}, {"version": "future"}, {"water": [1]}, {"tick": True}):
        with pytest.raises(ValueError):
            Field(41, {**before, **changed})


def test_service_rechecks_permissions_and_never_commits_before_record_sync():
    world = load_world(RELEASE)
    config = copy.deepcopy(CONFIG)
    service = Service(config, world)
    before = copy.deepcopy(service.field.state)

    class BrokenRecorder:
        def record(self, **event):
            pass

        def sync(self):
            raise OSError("simulated record failure")

    recorder = BrokenRecorder()
    config["capabilities"] = {"deny": ["world.act"]}
    assert service.execute("world.act", {"x": 0, "y": 0, "drive": 3}, recorder,
                           {"kind": "supervisor"}, "call-1")["status"] == "denied"
    config["capabilities"] = {}
    with pytest.raises(OSError, match="record failure"):
        service.execute("world.act", {"x": 0, "y": 0, "drive": 3}, recorder,
                        {"kind": "supervisor"}, "call-1")
    assert service.field.state == before


def test_demo_replays_and_matches_uninterrupted_across_resume(field_root, backend):  # noqa: F811
    reference = run_experiment(DEMO, field_root)
    paused = run_experiment(DEMO, field_root, pause_after_steps=5)
    assert reference.state == "completed" and paused.state == "paused"
    before = (paused.archive / "events.jsonl").read_bytes()
    resumed = resume_experiment(paused.archive, field_root)
    assert resumed.state == "completed", resumed.reason
    assert (resumed.archive / "events.jsonl").read_bytes().startswith(before)
    a, b = results(reference.archive), results(resumed.archive)
    assert all(r["status"] == "ok" for r in b), b
    assert [r["result"] for r in a] == [r["result"] for r in b]
    assert b[4]["result"] == b[5]["result"]
    assert b[7]["result"] == b[8]["result"]
    assert b[9]["result"]["outcome"]["result"] == b[8]["result"]
    control = Field(41)
    control.apply("world.advance", {"ticks": 4})
    affected = b[4]["result"]
    assert affected["cells"][14]["water"] > control.observe()["cells"][14]["water"]
    assert affected["cells"][15]["water"] > control.observe()["cells"][15]["water"]
    assert checkpoint_state(reference.archive)["phenomena"] == checkpoint_state(
        resumed.archive)["phenomena"]
    assert verify_archive(reference.archive) == verify_archive(resumed.archive) == []
    assert backend.running(resumed.run_id) == []
    events = events_of(resumed.archive)
    altered = copy.deepcopy(events)
    call_event = next(e for e in altered if e["payload"].get("extensions", {}).get(
        "stage") == "phenomena_call")
    call_event["payload"]["extensions"]["after"] = "0" * 64
    assert replay(CONFIG, altered)[0]
    assert replay(CONFIG, events)[0] == []


def test_workspace_forgeries_and_agent_code_cannot_rewrite_the_world(
        field_root, tmp_path, backend):  # noqa: F811
    code = '''import json, os
from pathlib import Path
assert not Path('/mnt/c').exists()
assert not Path('/world/phenomena.py').exists()
assert not Path('/world/capabilities/world.advance/implementation.py').exists()
assert not any(k.startswith('OPEN_LAB') for k in os.environ)
for name in ('phenomena.json', 'runtime_state.json', 'world-state.json'):
    Path(name).write_text(json.dumps({'seed': 0, 'water': [999999]*36}))
print('workspace forgeries written')
'''
    path = configured(tmp_path, [
        {"tool_calls": [call("world.observe")]},
        {"tool_calls": [call("python.run", code=code)]},
        {"tool_calls": [call("world.observe")]},
        {"tool_calls": [call("world.act", x=0, y=0, drive=4)]},
        {"tool_calls": [call("world.observe")]},
    ])
    result = run_experiment(path, field_root)
    assert result.state == "completed", result.reason
    found = results(result.archive)
    assert found[1]["result"]["exit_code"] == 0, found[1]
    assert found[0]["result"] == found[2]["result"] == found[4]["result"]
    assert found[3]["status"] == "error"
    assert checkpoint_state(result.archive)["phenomena"] == Field(41).state
    assert verify_archive(result.archive) == []


def test_crash_after_recorded_operation_rolls_back_to_the_complete_checkpoint(
        field_root, tmp_path, backend, monkeypatch):  # noqa: F811
    script = tmp_path / "crash_before_barrier.py"
    script.write_text('''import os
from open_lab_simulation import worker
original = worker.Channel.state
def interrupted(self, state, final=False):
    if state['steps'] == 1:
        os._exit(23)
    return original(self, state, final)
worker.Channel.state = interrupted
worker.main()
''', encoding="utf-8")
    path = configured(tmp_path, [
        {"tool_calls": [call("world.advance", ticks=3)]},
        {"tool_calls": [call("world.observe")]},
    ])
    with monkeypatch.context() as patch:
        patch.setattr(supervisor, "WORKER_COMMAND", [sys.executable, str(script)])
        failed = run_experiment(path, field_root)
    assert failed.state == "failed", failed.reason
    assert results(failed.archive)[0]["result"]["tick"] == 3
    assert checkpoint_state(failed.archive)["phenomena"]["tick"] == 0
    assert verify_archive(failed.archive) == []
    before = (failed.archive / "events.jsonl").read_bytes()
    resumed = resume_experiment(failed.archive, field_root)
    assert resumed.state == "completed", resumed.reason
    assert (resumed.archive / "events.jsonl").read_bytes().startswith(before)
    assert [r["result"]["tick"] for r in results(resumed.archive)] == [3, 3, 3]
    assert checkpoint_state(resumed.archive)["phenomena"]["tick"] == 3
    assert verify_archive(resumed.archive) == []


@pytest.mark.parametrize("seed", [True, -1, 2**32, "41"])
def test_configuration_is_bounded_and_not_prompt_authority(seed):
    config = copy.deepcopy(CONFIG)
    config["extensions"]["phenomena"] = {"seed": seed}
    with pytest.raises(ValueError, match="seed"):
        policy(config, load_world(RELEASE))


def test_disabled_and_denied_world_operations_are_hidden_or_refused(
        field_root, tmp_path, backend):  # noqa: F811
    replies = [{"tool_calls": [{"capability_id": "capabilities.search",
                                "arguments": {"query": "world field"}}]},
               {"tool_calls": [call("world.observe")]}]
    config = copy.deepcopy(CONFIG)
    config["extensions"].pop("phenomena")
    config["extensions"]["scripted_model"]["replies"] = replies
    path = tmp_path / "disabled.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    result = run_experiment(path, field_root)
    found = results(result.archive)
    assert not any(m["id"] in phenomena.IDS for m in found[0]["result"]["matches"])
    assert found[1]["status"] == "denied"
    assert verify_archive(result.archive) == []
    denied = configured(tmp_path, [{"tool_calls": [call("world.act", x=0, y=0, drive=3)]},
                                   {"tool_calls": [call("world.observe")]}],
                        capabilities={"deny": ["world.act"]})
    result = run_experiment(denied, field_root)
    assert results(result.archive)[0]["status"] == "denied"
    assert results(result.archive)[1]["result"] == Field(41).observe()
    assert verify_archive(result.archive) == []


def test_missing_reserved_world_capabilities_refuse_launch(root, make_config):
    path = make_config([], extensions={"phenomena": {"seed": 41}})
    with pytest.raises(LaunchError, match="container-tier"):
        run_experiment(path, root)


def test_pipe_binding_duplicate_refusal_and_worker_state_cannot_forge_private_snapshot(
        field_root, tmp_path, backend, monkeypatch):  # noqa: F811
    script = tmp_path / "hostile_worker.py"
    script.write_text('''import json, sys
from open_lab_simulation.policies import digest
spec = json.loads(sys.stdin.readline())
assert spec['phenomena'] is True
prior = spec.get('runtime_state', {})
assert 'phenomena' not in prior
def send(message):
    print(json.dumps(message), flush=True)
    return json.loads(sys.stdin.readline())
unbound = send({'world': {'id': 'world.observe', 'arguments': {}}})['world_result']
assert unbound['status'] == 'denied'
n = prior.get('call_count', 0) + 1
cid = f'call-{n:06d}'
ident = 'world.observe' if prior else 'world.act'
args = {} if prior else {'x': 1, 'y': 1, 'drive': 3}
send({'event': {'recorder': 'tool_host', 'type': 'tool.call', 'actor': {'kind': 'agent'},
               'correlation_id': cid, 'payload': {'capability_id': ident, 'arguments': args}}})
wrong = send({'world': {'id': 'world.advance', 'arguments': {'ticks': 1}}})['world_result']
assert wrong['status'] == 'denied'
outcome = send({'world': {'id': ident, 'arguments': args}})['world_result']
assert outcome['status'] == 'ok'
assert send({'world': {'id': ident, 'arguments': args}})['world_result']['status'] == 'denied'
send({'event': {'recorder': 'tool_host', 'type': 'tool.result', 'actor': {'kind': 'system'},
               'correlation_id': cid, 'payload': outcome}})
state = {'messages': prior.get('messages', [{'role':'system','content':'test'},
                                          {'role':'user','content':'test'}]),
         'steps': prior.get('steps', 0) + 1, 'script_cursor': 0, 'call_count': n, 'tokens_used': 0,
         'policy_sha256': digest(spec['model']['runtime_policy']),
         'policy_state': {'idle': False, 'tool_free_replies': 0},
         'phenomena': {'forged': True}}
ack = send({'state': state, 'final': True})
print(json.dumps({'end': {'reason': 'operator_pause' if ack.get('stop') else 'model_finished',
                         'steps': state['steps']}}), flush=True)
''', encoding="utf-8")
    monkeypatch.setattr(supervisor, "WORKER_COMMAND", [sys.executable, str(script)])
    paused = run_experiment(DEMO, field_root, pause_after_steps=1)
    assert paused.state == "paused", paused.notes
    expected = Field(41)
    expected.apply("world.act", {"x": 1, "y": 1, "drive": 3})
    assert checkpoint_state(paused.archive)["phenomena"] == expected.state
    resumed = resume_experiment(paused.archive, field_root)
    assert resumed.state == "completed", resumed.notes
    assert checkpoint_state(resumed.archive)["phenomena"] == expected.state
    assert verify_archive(resumed.archive) == []
    operations = [e for e in events_of(resumed.archive) if e["payload"].get("extensions", {}).get(
        "stage") == "phenomena_call"]
    assert len(operations) == 2
    assert digest(expected.state) == operations[-1]["payload"]["extensions"]["after"]
