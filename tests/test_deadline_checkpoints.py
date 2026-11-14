"""Supervisor-owned graceful deadlines, using real worker/capability processes."""

import json
import sqlite3
import time
import zipfile
from io import BytesIO
from types import SimpleNamespace

import pytest

from conftest import call, events_of
from open_lab_simulation import supervisor
from open_lab_simulation.archive import verify_archive
from open_lab_simulation.checkpoints import CheckpointTimeout, checkpoint_events
from open_lab_simulation.supervisor import LaunchError, resume_experiment, run_experiment


def metadata(result):
    event = checkpoint_events(events_of(result.archive))[-1]
    folder = result.archive / "checkpoints" / event["payload"]["checkpoint_id"]
    return folder, json.loads((folder / "checkpoint.json").read_text())


def assert_deadline(result):
    assert (result.state, result.reason, result.notes) == ("terminated", "deadline", [])
    actions = [e["payload"] for e in events_of(result.archive) if e["type"] == "supervisor.action"]
    assert actions[-1]["action"] == "terminated"
    assert sum(a.get("extensions", {}).get("stage") == "deadline_checkpoint_requested"
               for a in actions) == 1
    assert verify_archive(result.archive) == []


def test_grace_saves_consistent_step_and_never_starts_following_step(root, make_config):
    result = run_experiment(make_config(
        [
            {"delay_s": 2.1, "tool_calls": [
                call("memory.write", key="plan", value="saved"),
                call("workspace.write", path="NOTES.md", text="saved"),
            ]},
            {"tool_calls": [call("workspace.write", path="too-late.txt", text="late")]},
        ],
        wall_clock_s=2,
        checkpointing={},
        termination={"on_deadline": "checkpoint_then_stop", "grace_period_s": 3},
    ), root)
    assert_deadline(result)
    folder, checkpoint = metadata(result)
    assert checkpoint["reason"] == "pre_shutdown"
    assert checkpoint["progress"]["step_count"] == 1
    state = json.loads((folder / "runtime_state.json").read_text())
    conversation = json.loads((folder / "conversation.json").read_text())
    assert state["script_cursor"] == state["steps"] == 1
    assert conversation[-1]["role"] == "tool"
    with zipfile.ZipFile(folder / "workspace.zip") as bundle:
        assert bundle.read("NOTES.md") == b"saved"
        assert "too-late.txt" not in bundle.namelist()
    with sqlite3.connect(folder / "memory.sqlite") as connection:
        assert connection.execute("SELECT key,value FROM notes").fetchall() == [("plan", "saved")]
    assert (result.workspace / "NOTES.md").read_text() == "saved"
    assert not (result.workspace / "too-late.txt").exists()
    assert sum(e["type"] == "model.call" for e in events_of(result.archive)) == 1
    with pytest.raises(LaunchError, match="exhausted"):
        resume_experiment(result.archive, root)


@pytest.mark.parametrize("busy_tool", [False, True])
def test_grace_expiry_kills_busy_model_or_capability_and_retains_initial_snapshot(
    root, make_config, busy_tool,
):
    reply = {"tool_calls": [call("test.heartbeat")]} if busy_tool else {"delay_s": 60}
    started = time.monotonic()
    result = run_experiment(make_config(
        [reply], wall_clock_s=2, checkpointing={},
        termination={"on_deadline": "checkpoint_then_stop", "grace_period_s": 1},
    ), root)
    duration = time.monotonic() - started
    assert 3 <= duration < 12
    assert_deadline(result)
    _, checkpoint = metadata(result)
    assert checkpoint["progress"]["step_count"] == 0
    assert len(checkpoint_events(events_of(result.archive))) == 1
    if busy_tool:
        heartbeat = result.workspace / "heartbeat.txt"
        previous = heartbeat.read_bytes()
        time.sleep(0.2)
        assert heartbeat.read_bytes() == previous


@pytest.mark.parametrize("termination", [
    {"on_deadline": "checkpoint_then_stop"},
    {"on_deadline": "checkpoint_then_stop", "grace_period_s": 0},
    {"on_deadline": "hard_stop", "grace_period_s": 30},
])
def test_zero_or_omitted_grace_and_hard_stop_do_not_extend_deadline(
    root, make_config, termination,
):
    started = time.monotonic()
    result = run_experiment(make_config(
        [{"delay_s": 60}], wall_clock_s=1, checkpointing={}, termination=termination,
    ), root)
    assert time.monotonic() - started < 8
    assert (result.state, result.reason) == ("terminated", "deadline")
    assert metadata(result)[1]["progress"]["step_count"] == 0
    if termination["on_deadline"] == "checkpoint_then_stop":
        assert_deadline(result)
    else:
        assert verify_archive(result.archive) == []


def test_snapshot_timeout_during_grace_keeps_previous_checkpoint(root, make_config, monkeypatch):
    original = supervisor.write_checkpoint

    def timeout(*args, **kwargs):
        if args[5]["steps"]:
            raise CheckpointTimeout("injected timeout while writing deadline snapshot")
        return original(*args, **kwargs)

    monkeypatch.setattr(supervisor, "write_checkpoint", timeout)
    result = run_experiment(make_config(
        [{"delay_s": 2.1}], wall_clock_s=2, checkpointing={},
        termination={"on_deadline": "checkpoint_then_stop", "grace_period_s": 3},
    ), root)
    assert_deadline(result)
    assert metadata(result)[1]["progress"]["step_count"] == 0
    assert len(checkpoint_events(events_of(result.archive))) == 1


def test_grace_deadline_uses_original_budget_after_resume(root, make_config):
    config = make_config(
        [{"text": "first"}, {"delay_s": 2.1}, {"text": "must not run"}],
        wall_clock_s=2, checkpointing={},
        termination={"on_deadline": "checkpoint_then_stop", "grace_period_s": 3},
    )
    paused = run_experiment(config, root, pause_after_steps=1)
    prior = (paused.archive / "events.jsonl").read_bytes()
    result = resume_experiment(paused.archive, root)
    assert_deadline(result)
    assert (result.archive / "events.jsonl").read_bytes().startswith(prior)
    assert metadata(result)[1]["progress"]["step_count"] == 2
    assert sum(e["type"] == "model.call" for e in events_of(result.archive)) == 2


def test_grace_mode_finishes_normally_before_deadline(root, make_config):
    result = run_experiment(make_config(
        [{"text": "done"}], checkpointing={},
        termination={"on_deadline": "checkpoint_then_stop", "grace_period_s": 3},
    ), root)
    assert result.state == "completed"
    assert metadata(result)[1]["reason"] == "pre_shutdown"
    actions = [e["payload"] for e in events_of(result.archive)
               if e["type"] == "supervisor.action"]
    assert actions and all(a.get("extensions", {}).get("stage") == "model_input"
                           for a in actions)
    assert verify_archive(result.archive) == []


def test_snapshot_crossing_nominal_deadline_does_not_release_worker(
    root, make_config, monkeypatch,
):
    original = supervisor.write_checkpoint

    def slow_snapshot(*args, **kwargs):
        time.sleep(1.1)
        return original(*args, **kwargs)

    monkeypatch.setattr(supervisor, "write_checkpoint", slow_snapshot)
    result = run_experiment(make_config(
        [{"text": "must not start"}], wall_clock_s=1, checkpointing={},
        termination={"on_deadline": "checkpoint_then_stop", "grace_period_s": 3},
    ), root)
    assert_deadline(result)
    assert metadata(result)[1]["progress"]["step_count"] == 0
    assert not any(e["type"] == "model.call" for e in events_of(result.archive))


def test_queued_barrier_after_kill_is_never_snapshotted():
    actions = []
    killed = []
    tree = SimpleNamespace(
        process=SimpleNamespace(stdout=BytesIO(b'{"state": {"steps": 0}}\n')),
        kill=lambda: killed.append(True),
    )

    def snapshot(*args):
        pytest.fail("a queued state barrier cannot snapshot a killed tree")

    result = supervisor._supervise(
        tree, SimpleNamespace(record=lambda **event: actions.append(event)),
        None, 0, {"kind": "supervisor"}, checkpointing={}, snapshot=snapshot,
        termination={"on_deadline": "checkpoint_then_stop"},
    )
    assert result == ("terminated", "deadline")
    assert killed == [True]
    assert [e["payload"]["action"] for e in actions] == ["limit_enforced", "terminated"]


def test_initial_snapshot_expiring_grace_still_records_deadline_request(
    root, make_config, monkeypatch,
):
    original = supervisor.write_checkpoint

    def delayed(*args, **kwargs):
        time.sleep(2.1)
        return original(*args, **kwargs)

    monkeypatch.setattr(supervisor, "write_checkpoint", delayed)
    result = run_experiment(make_config(
        [{"text": "must not start"}], wall_clock_s=1, checkpointing={},
        termination={"on_deadline": "checkpoint_then_stop", "grace_period_s": 1},
    ), root)
    assert_deadline(result)
    assert checkpoint_events(events_of(result.archive)) == []
    assert not any(e["type"] == "model.call" for e in events_of(result.archive))
