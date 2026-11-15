"""S011a: the registered runtime-kill reliability fault and its recovery.

The supervisor itself kills the runtime once at a registered active second; the
run's own recovery policy must restore it. Scripted and fake-provider models only.
"""

from __future__ import annotations

import json

import pytest

from conftest import call, events_of
from open_lab_simulation import faults, recovery
from open_lab_simulation.archive import verify_archive
from open_lab_simulation.supervisor import LaunchError, run_experiment
from test_hosted_compatibility import (  # noqa: F401 - chatgpt_run is a fixture
    ANSWER,
    PROVIDER,
    SEARCH,
    chatgpt_run,
)

FAULT = {"id": faults.RUNTIME_KILL, "at_active_s": 1}
RECOVERY = {"max_restarts": 1, "auto_resume": True}


@pytest.fixture
def hosted_run(chatgpt_run):  # noqa: F811 - the shared compatibility harness
    return chatgpt_run


def stages(archive):
    return [recovery.event_extension(e).get("stage") for e in events_of(archive)]


def test_the_fault_fires_once_and_the_run_recovers(root, make_config):
    replies = [{"delay_s": 0.6, "tool_calls": [call("workspace.write", path=f"n{i}.txt",
                                                    text=str(i))]} for i in range(4)]
    config = make_config(replies, checkpointing={"interval_s": 1, "keep_last": 4},
                         limits={"max_steps": 8, "wall_clock_s": 30},
                         extensions={"scripted_model": {"replies": replies},
                                     "recovery": RECOVERY, "reliability_fault": FAULT})
    result = run_experiment(config, root)
    found = stages(result.archive)
    assert found.count(faults.RUNTIME_KILL_STAGE) == 1
    assert found.index(faults.RUNTIME_KILL_STAGE) < found.index("recovery_restart")
    assert recovery.restarts_used(events_of(result.archive)) == 1
    assert result.state == "completed"
    assert (result.workspace / "n3.txt").read_text() == "3"
    assert verify_archive(result.archive) == []


def test_a_hosted_run_killed_while_idle_stays_idle_after_recovery(hosted_run):
    quiet = {"id": "quiet-until-deadline-v1", "max_tool_free_replies": 1}
    result, upstream = hosted_run(
        [SEARCH, ANSWER, ANSWER], max_steps=10, wall_clock_s=8,
        checkpointing={"interval_s": 1, "keep_last": 4},
        extensions={"runtime_policy": quiet,
                    "recovery": {**RECOVERY, "hosted_identity": recovery.HOSTED_IDENTITY},
                    "reliability_fault": {"id": faults.RUNTIME_KILL, "at_active_s": 4}})
    found = stages(result.archive)
    assert found.index("runtime_idle") < found.index(faults.RUNTIME_KILL_STAGE)
    assert "recovery_restart" in found and "hosted_identity_pin" in found
    assert len(upstream.requests) == 2  # recovery restored the idle latch: no new request
    summary = json.loads((result.archive / "summary.json").read_text(encoding="utf-8"))
    assert summary["final_state"] == "terminated" and summary["reason"] == "deadline"
    assert verify_archive(result.archive) == []


@pytest.mark.parametrize("change, message", [
    ({"reliability_fault": {"id": "other", "at_active_s": 1}}, "unsupported reliability"),
    ({"reliability_fault": {**FAULT, "at_active_s": 30}}, "inside the wall clock"),
    ({"reliability_fault": {**FAULT, "at_active_s": True}}, "inside the wall clock"),
    ({"reliability_fault": FAULT, "recovery": {"max_restarts": 1}}, "automatic recovery"),
    ({"reliability_fault": FAULT, "recovery": {**RECOVERY, "max_restarts": 0}},
     "automatic recovery"),
])
def test_invalid_fault_configurations_are_refused_before_launch(root, make_config, change,
                                                                message):
    extensions = {"scripted_model": {"replies": []}, "recovery": RECOVERY, **change}
    config = make_config([], checkpointing={}, limits={"wall_clock_s": 30},
                         extensions=extensions)
    with pytest.raises(LaunchError, match=message):
        run_experiment(config, root)
    assert not (root / "runs").exists()


def test_provider_id_is_the_hosted_harness():
    assert PROVIDER == "openai-chatgpt"


def test_s011a_configs_differ_from_the_s011p_hour_only_where_registered():
    from conftest import REPO
    base = json.loads((REPO / "experiments/s011p/hour-001.json").read_text(encoding="utf-8"))
    for n, at in ((1, 300), (2, 900), (3, 1800)):
        config = json.loads((REPO / f"experiments/s011a/hour-{n}.json").read_text(
            encoding="utf-8"))
        extensions = config["extensions"]
        assert extensions.pop("reliability_fault") == {"id": faults.RUNTIME_KILL,
                                                       "at_active_s": at}
        assert extensions.pop("recovery") == {**RECOVERY,
                                              "hosted_identity": recovery.HOSTED_IDENTITY}
        assert faults.runtime_kill_at({**config, "extensions": {
            "recovery": {**RECOVERY}, "reliability_fault": {"id": faults.RUNTIME_KILL,
                                                            "at_active_s": at}}}) == at
        for key in ("experiment_id", "title", "research_question", "tags"):
            config.pop(key), base.get(key)
        assert config == {k: v for k, v in base.items()
                          if k not in ("experiment_id", "title", "research_question", "tags")}
