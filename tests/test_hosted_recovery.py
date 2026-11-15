"""S011r / ADR-0021: opted-in hosted runs recover from Laboratory faults only.

Full supervised runs against a fake Responses endpoint: the real supervisor,
broker, adapter, worker and checkpoints. Faults are injected by killing the
runtime process, as in test_recovery. No real provider is contacted.
"""

from __future__ import annotations

import json

import pytest

from conftest import events_of
from open_lab_simulation import providers, recovery
from open_lab_simulation.archive import verify_archive
from open_lab_simulation.supervisor import LaunchError, resume_experiment
from test_hosted_compatibility import (  # noqa: F401 - chatgpt_run is a fixture
    ANSWER,
    DESCRIBE,
    INVOKE,
    PROVIDER,
    SEARCH,
    call,
    chatgpt_run,
    error,
    errors_of,
    sse,
    text,
)
from test_recovery import crash_workers

OPT_IN = {"recovery": {"max_restarts": 1, "auto_resume": True,
                       "hosted_identity": recovery.HOSTED_IDENTITY}}
CHECKPOINTS = {"interval_s": 1, "keep_last": 4}


@pytest.fixture
def hosted_run(chatgpt_run):  # noqa: F811 - the shared compatibility harness
    return chatgpt_run


def reply(version, *output):
    return (200, sse({"type": "response.created", "response": {}}, {
        "type": "response.completed", "response": {
            "model": version, "output": list(output),
            "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}}}))


def stages(archive):
    return [recovery.event_extension(e).get("stage") for e in events_of(archive)]


def ledger(root):
    return providers.Ledger(root, PROVIDER).entries()


def test_a_killed_runtime_recovers_under_one_run_and_one_pinned_version(
        hosted_run, monkeypatch):
    threads, errors = crash_workers(monkeypatch, hosted_run.root)
    result, upstream = hosted_run([SEARCH, DESCRIBE, INVOKE, ANSWER] * 3, max_steps=6,
                                   checkpointing=CHECKPOINTS, extensions=OPT_IN)
    for thread in threads:
        thread.join()
    assert errors == []
    events = events_of(result.archive)
    assert recovery.restarts_used(events) == 1
    pins = [recovery.event_extension(e) for e in events
            if recovery.event_extension(e).get("stage") == "hosted_identity_pin"]
    assert len(pins) == 1 and pins[0]["policy"] == recovery.HOSTED_IDENTITY
    assert pins[0]["model_version"] in ("gpt-compat-1", None)
    reserves = [e for e in ledger(hosted_run.root) if e["kind"] == "reserve"]
    assert {e["run_id"] for e in reserves} == {result.run_id}
    assert len(reserves) == len(upstream.requests)  # replayed requests are counted
    model = json.loads((result.archive / "model.json").read_text(encoding="utf-8"))
    assert "not cryptographically attested" in model["identity"]["identity_limit"]
    assert model["identity"]["digest"] is None
    assert verify_archive(result.archive) == []


def test_a_different_reported_version_is_withheld_and_stops_the_run(hosted_run):
    other = reply("gpt-other-2", call("capabilities_describe", {"id": "math.evaluate"},
                                      "call_2"))
    result, upstream = hosted_run([SEARCH, other, ANSWER], max_steps=4,
                                   checkpointing=CHECKPOINTS, extensions=OPT_IN)
    assert result.state == "failed"
    events = events_of(result.archive)
    # The second model's tool call never reached the runtime.
    assert [e["payload"]["capability_id"] for e in events if e["type"] == "tool.call"] == [
        "capabilities.search"]
    outcomes = [e for e in ledger(hosted_run.root) if e["kind"] == "outcome"]
    assert outcomes[-1]["result"] == "terminal"
    assert outcomes[-1]["code"] == "provider_identity_changed:gpt-other-2"
    assert len(upstream.requests) == 2
    assert recovery.restarts_used(events) == 0  # a provider stop is never recovered
    with pytest.raises(LaunchError, match="provider"):
        resume_experiment(result.archive, hosted_run.root)
    assert len(upstream.requests) == 2
    assert verify_archive(result.archive) == []


def test_a_usage_limit_refusal_is_not_recovered_or_resumable(hosted_run):
    refusal = error(429, "subscription_sharing_usage_limit_exceeded")
    result, upstream = hosted_run([SEARCH, refusal, ANSWER], max_steps=4,
                                   checkpointing=CHECKPOINTS, extensions=OPT_IN)
    assert result.state == "failed"
    assert recovery.restarts_used(events_of(result.archive)) == 0
    assert any("laboratory_provider_terminal" in m for m in errors_of(result))
    with pytest.raises(LaunchError, match="provider"):
        resume_experiment(result.archive, hosted_run.root)
    assert len(upstream.requests) == 2


def test_an_unavailable_provider_is_not_recovered_automatically(hosted_run):
    down = error(503, "server_error")
    result, upstream = hosted_run([SEARCH, down, down, ANSWER], max_steps=4, attempts=2,
                                   checkpointing=CHECKPOINTS, extensions=OPT_IN)
    assert result.state == "failed"
    assert recovery.restarts_used(events_of(result.archive)) == 0
    assert len(upstream.requests) == 3


def test_operator_resume_keeps_the_pin_and_refuses_a_changed_broker_profile(hosted_run):
    result, upstream = hosted_run([SEARCH, DESCRIBE, INVOKE, ANSWER], max_steps=4,
                                   pause_after_steps=1, checkpointing=CHECKPOINTS,
                                   extensions={"recovery": {
                                       "max_restarts": 0,
                                       "hosted_identity": recovery.HOSTED_IDENTITY}})
    summary = json.loads((result.archive / "summary.json").read_text(encoding="utf-8"))
    assert summary["final_state"] == "paused"
    sent = len(upstream.requests)
    profile = providers.load_profile(hosted_run.root, PROVIDER)
    providers.write_profile(hosted_run.root, PROVIDER, enabled=True, per_run=50, per_day=500,
                            total=500, request_timeout_s=profile["request_timeout_s"],
                            max_attempts_per_step=profile["max_attempts_per_step"],
                            min_interval_s=5)
    with pytest.raises(LaunchError, match="broker profile"):
        resume_experiment(result.archive, hosted_run.root)
    assert len(upstream.requests) == sent
    providers.write_profile(hosted_run.root, PROVIDER, enabled=True, per_run=50, per_day=500,
                            total=500, request_timeout_s=profile["request_timeout_s"],
                            max_attempts_per_step=profile["max_attempts_per_step"],
                            min_interval_s=profile["min_interval_s"])
    resumed = resume_experiment(result.archive, hosted_run.root)
    assert resumed.run_id == result.run_id
    pin = next(recovery.event_extension(e) for e in events_of(resumed.archive)
               if recovery.event_extension(e).get("stage") == "hosted_identity_pin")
    assert pin["model_version"] == "gpt-compat-1"
    assert verify_archive(resumed.archive) == []


def test_identity_policy_values_are_closed():
    config = {"checkpointing": {}, "extensions": {"recovery": {
        "max_restarts": 1, "hosted_identity": "anything-else"}}}
    with pytest.raises(ValueError, match="hosted_identity"):
        recovery.policy(config)


def test_an_unreported_version_cannot_establish_a_pin():
    broker = providers.Broker.__new__(providers.Broker)
    broker.pinned_version = None
    outcome = broker._check_identity(providers.Outcome("ok", "ok", reply={}))
    assert outcome.kind == "terminal"
    assert outcome.code == "provider_identity_changed:unreported"


def test_text_helper_is_shared():
    # Keeps the imported wire helpers honest if the compatibility module changes.
    assert text("x")["content"][0]["text"] == "x"
