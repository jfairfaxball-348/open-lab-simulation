"""Policy treatments, exact model inputs and idle checkpoint/deadline authority."""

import json
import threading
import time

import pytest

from conftest import call, events_of
from open_lab_simulation import policies, recovery, supervisor
from open_lab_simulation.archive import verify_archive
from open_lab_simulation.checkpoints import checkpoint_events
from open_lab_simulation.model import ModelError, OpenAIChatModel
from open_lab_simulation.supervisor import LaunchError, resume_experiment, run_experiment
from test_recovery import wait_until

pytest_plugins = ["test_model"]


def staged(archive, stage):
    return [e for e in events_of(archive)
            if e["payload"].get("extensions", {}).get("stage") == stage]


def continuations(archive):
    return [e["payload"]["text"] for e in events_of(archive)
            if e["type"] == "environment.message"
            and e["payload"]["kind"] == "continuation"]


def quiet(make_config, replies, **changes):
    return make_config(replies, extensions={"runtime_policy": {"id": policies.QUIET}}, **changes)


@pytest.mark.parametrize("raw", [
    {"id": "unknown-v1"}, {"turn_semantics": "assistant-only"},
    {"memory_policy": "automatic-summary"}, {"context_policy": "silent-trim"},
    {"unrecognised": 1}, {"max_tool_free_replies": 2},
    {"context_guard_tokens": True}, {"context_guard_tokens": 0},
    {"id": policies.QUIET, "max_tool_free_replies": 0},
    {"id": policies.QUIET, "max_tool_free_replies": 17},
    {"id": policies.QUIET, "max_tool_free_replies": True},
    {"id": policies.QUIET, "wake_after_s": 1},
])
def test_unsupported_policy_is_refused_before_run(root, make_config, raw):
    with pytest.raises(LaunchError):
        run_experiment(make_config([], extensions={"runtime_policy": raw}), root)
    assert not list((root / "runs").glob("*/*"))


def test_explicit_legacy_preserves_conversation_and_tool_inputs(root, make_config):
    replies = [{"text": "I am waiting."}, {"tool_calls": [call("text.stats", text="a b")]},
               {"text": "Done."}]
    default = run_experiment(make_config(replies, name="implicit"), root)
    explicit = run_experiment(make_config(replies, name="explicit",
        extensions={"runtime_policy": {"id": policies.LEGACY}}), root)
    assert continuations(default.archive) == continuations(explicit.archive)
    implicit_inputs = [e["payload"]["extensions"]["request"]
                       for e in staged(default.archive, "model_input")]
    explicit_inputs = [e["payload"]["extensions"]["request"]
                       for e in staged(explicit.archive, "model_input")]
    assert implicit_inputs == explicit_inputs
    assert len(continuations(default.archive)) == 2
    for result in (default, explicit):
        assert result.state == "completed"
        assert verify_archive(result.archive) == []
        assert json.loads((result.archive / "policy.json").read_text())["id"] == policies.LEGACY


def test_quiet_has_no_model_busy_loop_or_unrecorded_wake(root, make_config):
    config = quiet(make_config, [{"text": "Continue forever"}, {
        "tool_calls": [call("workspace.write", path="unexpected.txt", text="no")]}],
        wall_clock_s=2)
    start = time.monotonic()
    result = run_experiment(config, root)
    assert (result.state, result.reason) == ("terminated", "deadline")
    assert 1 <= time.monotonic() - start < 8
    assert len(staged(result.archive, "model_input")) == 1
    assert not continuations(result.archive)
    assert len(staged(result.archive, "runtime_idle")) == 1
    assert not (result.workspace / "unexpected.txt").exists()
    assert verify_archive(result.archive) == []


def test_reflection_threshold_and_parsed_call_reset(root, make_config):
    config = make_config([
        {"text": "one"}, {"tool_calls": [call("missing.tool")]},
        {"text": "two"}, {"text": "three"}, {"text": "must not run"},
    ], wall_clock_s=2, extensions={"runtime_policy": {
        "id": policies.QUIET, "max_tool_free_replies": 2}})
    result = run_experiment(config, root)
    assert len(staged(result.archive, "model_input")) == 4
    assert len(continuations(result.archive)) == 2
    idle = staged(result.archive, "runtime_idle")[0]["payload"]["extensions"]
    assert idle["policy_state"] == {"idle": True, "tool_free_replies": 2}
    assert verify_archive(result.archive) == []


def test_step_budget_stops_instead_of_waiting(root, make_config):
    result = run_experiment(quiet(make_config, [{"text": "idle"}],
        limits={"max_steps": 1}), root)
    assert (result.state, result.reason) == ("terminated", "max_steps")
    assert verify_archive(result.archive) == []


def test_graceful_deadline_checkpoints_while_idle(root, make_config):
    result = run_experiment(quiet(make_config, [{"text": "idle"}],
        wall_clock_s=1, checkpointing={"interval_s": 60, "keep_last": 4},
        termination={"on_deadline": "checkpoint_then_stop", "grace_period_s": 2}), root)
    assert (result.state, result.reason) == ("terminated", "deadline")
    event = checkpoint_events(events_of(result.archive))[-1]
    path = result.archive / "checkpoints" / event["payload"]["checkpoint_id"]
    assert json.loads((path / "runtime_state.json").read_text())["policy_state"]["idle"]
    assert json.loads((path / "checkpoint.json").read_text())["reason"] == "pre_shutdown"
    assert len(staged(result.archive, "model_input")) == 1
    assert verify_archive(result.archive) == []


def test_resume_preserves_idle_counter_policy_and_active_budget(root, make_config):
    config = quiet(make_config, [{"text": "idle"}, {"text": "must not run"}],
        wall_clock_s=2, checkpointing={"keep_last": 5})
    paused = run_experiment(config, root, pause_after_steps=1)
    assert paused.state == "paused"
    prefix = (paused.archive / "events.jsonl").read_bytes()
    time.sleep(0.1)
    result = resume_experiment(paused.archive, root)
    assert (result.state, result.reason) == ("terminated", "deadline")
    assert (result.archive / "events.jsonl").read_bytes().startswith(prefix)
    assert len(staged(result.archive, "model_input")) == 1
    assert staged(result.archive, "runtime_idle")[-1]["payload"]["extensions"]["restored"]
    assert verify_archive(result.archive) == []


def test_worker_recovery_does_not_wake_a_checkpointed_idle_run(root, make_config, monkeypatch):
    original = supervisor.ProcessTree
    threads, errors = [], []

    class CrashingTree(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if threads:
                return

            def crash():
                try:
                    def idle_checkpoint():
                        for a in (root / "runs").glob("*/*"):
                            for e in checkpoint_events(events_of(a)):
                                p = a / "checkpoints" / e["payload"]["checkpoint_id"]
                                if json.loads((p / "runtime_state.json").read_text()).get(
                                    "policy_state", {}).get("idle"):
                                    return True
                        return False
                    wait_until(idle_checkpoint)
                    self.process.kill()
                except Exception as error:
                    errors.append(error)
            thread = threading.Thread(target=crash)
            threads.append(thread)
            thread.start()

    monkeypatch.setattr(supervisor, "ProcessTree", CrashingTree)
    config = make_config([{"text": "idle"}, {"text": "must not run"}], wall_clock_s=4,
        checkpointing={"interval_s": 1, "keep_last": 8}, extensions={
            "runtime_policy": {"id": policies.QUIET}, "recovery": {"max_restarts": 1}})
    result = run_experiment(config, root)
    for thread in threads:
        thread.join()
    assert not errors
    assert result.reason == "deadline"
    assert recovery.restarts_used(events_of(result.archive)) == 1
    assert len(staged(result.archive, "model_input")) == 1
    assert verify_archive(result.archive) == []


def test_policy_config_and_archive_tampering_are_detected(root, make_config):
    paused = run_experiment(quiet(make_config, [{"text": "idle"}], checkpointing={}),
                            root, pause_after_steps=1)
    policy_path = paused.archive / "policy.json"
    policy_path.write_text("{}")
    assert any("policy" in p for p in verify_archive(paused.archive))
    with pytest.raises(LaunchError):
        resume_experiment(paused.archive, root)


def test_exact_http_requests_record_truncation_overflow_retries_and_failures(
    model_root, model_config, server,
):
    server.replies = [500, {"content": "reply"}, 400]
    result = run_experiment(model_config(extensions={"runtime_policy": {
        "context_guard_tokens": 1}}, limits={"max_steps": 3}), model_root)
    inputs = staged(result.archive, "model_input")
    assert len(inputs) == len(server.requests) == 3
    assert [e["payload"]["extensions"]["request"] for e in inputs] == server.requests
    assert [e["payload"]["extensions"]["attempt"] for e in inputs] == [1, 2, 1]
    assert all(e["payload"]["extensions"]["context"]["protected_overflow"] for e in inputs)
    assert all(r["messages"][0]["role"] == "system" for r in server.requests)
    assert all(r["messages"][1]["content"] == "You may use whatever you find."
               for r in server.requests)
    assert result.state == "failed"
    assert verify_archive(result.archive) == []


def test_unacknowledged_input_is_never_sent(server):
    def refused(payload):
        raise ModelError("input rejected")
    model = OpenAIChatModel("test", server.base + "/v1", "test", [], record_input=refused)
    with pytest.raises(ModelError, match="input rejected"):
        model.chat([{"role": "user", "content": "test"}])
    assert not server.requests


def test_memory_enters_input_only_after_explicit_retrieval(root, make_config):
    result = run_experiment(make_config([
        {"tool_calls": [call("memory.write", key="idea", value="unique memory value")]},
        {"tool_calls": [call("memory.search", query="unique")]},
        {"text": "done"},
    ]), root)
    inputs = [e["payload"]["extensions"]["request"] for e in staged(result.archive, "model_input")]
    # A write acknowledgement does not contain the note body as retrieved data.
    tools_before = [m["content"] for m in inputs[1]["messages"] if m["role"] == "tool"]
    tools_after = [m["content"] for m in inputs[2]["messages"] if m["role"] == "tool"]
    assert not any("unique memory value" in m for m in tools_before)
    assert any("unique memory value" in m for m in tools_after)
    assert verify_archive(result.archive) == []
