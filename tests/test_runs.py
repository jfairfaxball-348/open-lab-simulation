"""Whole runs: launch, supervision, recording, artifacts and refusal to launch."""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import pytest

from conftest import REPO, call, events_of, tree
from open_lab_simulation import supervisor
from open_lab_simulation.archive import verify_archive
from open_lab_simulation.artifacts import read_index
from open_lab_simulation.cli import main
from open_lab_simulation.contracts import Contracts, validate_event_stream
from open_lab_simulation.supervisor import LaunchError, run_experiment
from open_lab_simulation.world import load_world

SMOKE = REPO / "experiments" / "scripted-smoke.json"


def types(events: list[dict]) -> list[str]:
    return [event["type"] for event in events]


def results(events: list[dict]) -> list[dict]:
    return [event["payload"] for event in events if event["type"] == "tool.result"]


# --- a complete scripted run -------------------------------------------------

def test_scripted_run_is_fully_recorded(root, release):
    result = run_experiment(SMOKE, root)
    assert (result.state, result.notes) == ("completed", [])

    # The log is judged by the schemas the world release carries, not by ours.
    lines = (result.archive / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert validate_event_stream(Contracts(release / "contracts"), lines) == []
    assert verify_archive(result.archive) == []

    events = events_of(result.archive)
    assert [e["seq"] for e in events] == list(range(len(events)))
    states = [e["payload"]["to"] for e in events if e["type"] == "run.state_changed"]
    assert states == ["created", "starting", "running", "completed"]
    assert types(events).count("agent.output") == 4
    assert [r["status"] for r in results(events)] == ["ok", "ok", "error", "error"]
    assert results(events)[0]["result"] == {"characters": 19, "words": 3, "lines": 2}
    assert results(events)[2]["error"] == "capability_error: division by zero"
    assert all(e["epistemic_status"] == "observed" for e in events)
    assert {e["recorder"] for e in events} == {
        "experiment_manager", "supervisor", "model_gateway", "agent_runtime", "tool_host"
    }


def test_archive_holds_frozen_configuration_world_and_contracts(root, release):
    result = run_experiment(SMOKE, root)
    assert (result.archive / "experiment.json").read_bytes() == SMOKE.read_bytes()
    assert (result.archive / "world-release.json").read_bytes() == (
        release / "world-release.json"
    ).read_bytes()
    assert tree(result.archive / "contracts") == tree(release / "contracts")
    summary = json.loads((result.archive / "summary.json").read_text(encoding="utf-8"))
    assert summary["final_state"] == "completed"
    assert summary["model"] == {"model_id": "scripted", "real_model": False}
    assert summary["event_count"] == result.events


def test_nothing_is_written_outside_the_workspace_and_archive(root, release, tmp_path):
    before_root, before_tmp = tree(root), tree(tmp_path)
    before_repo = tree(REPO / "experiments") | tree(REPO / "src")
    result = run_experiment(SMOKE, root)
    new = tree(root) - before_root
    archive = result.archive.relative_to(root).as_posix()
    workspace = result.workspace.relative_to(root).as_posix()
    assert new and all(path.startswith((archive + "/", workspace + "/")) for path in new)
    assert tree(tmp_path) - before_tmp == {f"data/{path}" for path in new}
    without_caches = {p for p in tree(REPO / "experiments") | tree(REPO / "src")
                      if "__pycache__" not in p}
    assert without_caches == {p for p in before_repo if "__pycache__" not in p}
    load_world(release)  # still matches its digest


def test_each_run_gets_a_fresh_workspace_and_its_own_archive(root):
    first = run_experiment(SMOKE, root)
    second = run_experiment(SMOKE, root)
    assert first.run_id != second.run_id
    assert first.archive != second.archive and first.workspace != second.workspace
    assert list(second.workspace.iterdir()) == []
    assert verify_archive(first.archive) == verify_archive(second.archive) == []


# --- the supervisor's limits ---------------------------------------------------

def test_run_past_its_deadline_is_terminated_by_the_supervisor(root, make_config):
    config = make_config([{"text": "thinking", "delay_s": 60}], wall_clock_s=1)
    started = time.monotonic()
    result = run_experiment(config, root)
    assert time.monotonic() - started < 20
    assert (result.state, result.reason) == ("terminated", "deadline")
    events = events_of(result.archive)
    action = next(e for e in events if e["type"] == "supervisor.action")
    assert action["recorder"] == "supervisor"
    assert action["payload"] == {"action": "terminated", "detail": "wall_clock_s limit reached"}
    states = [e["payload"]["to"] for e in events if e["type"] == "run.state_changed"]
    assert states == ["created", "starting", "running", "stopping", "terminated"]
    assert verify_archive(result.archive) == []


def test_deadline_kills_a_capability_still_running(root, make_config):
    config = make_config([{"text": "start", "tool_calls": [call("test.heartbeat")]}],
                         wall_clock_s=2)
    result = run_experiment(config, root)
    assert result.state == "terminated"
    heartbeat = result.workspace / "heartbeat.txt"
    assert heartbeat.is_file(), "the capability really was running"
    seen = heartbeat.read_text()
    time.sleep(1)
    assert heartbeat.read_text() == seen, "the capability process outlived the run"
    # Its unfinished call is in the record, and so is the file it left behind.
    events = events_of(result.archive)
    assert types(events).count("tool.call") == 1 and types(events).count("tool.result") == 0
    assert any(e["type"].startswith("artifact.") for e in events)
    assert verify_archive(result.archive) == []


def test_step_limit_ends_the_run(root, make_config):
    config = make_config([{"text": str(n)} for n in range(5)], limits={"max_steps": 2})
    result = run_experiment(config, root)
    assert (result.state, result.reason) == ("terminated", "max_steps")
    events = events_of(result.archive)
    assert types(events).count("agent.output") == 2
    assert [e["payload"]["action"] for e in events if e["type"] == "supervisor.action"] == [
        "limit_enforced"
    ]


# --- artifacts -----------------------------------------------------------------

def test_file_written_by_a_capability_is_recorded_and_preserved(root, make_config):
    config = make_config([
        {"text": "write", "tool_calls": [call("file.write", path="notes/plan.md", content="one")]},
        {"text": "revise", "tool_calls": [call("file.write", path="notes/plan.md",
                                               content="two")]},
        {"text": "remove", "tool_calls": [call("file.write", path="notes/plan.md",
                                               content=None)]},
    ])
    result = run_experiment(config, root)
    assert result.state == "completed"
    events = events_of(result.archive)
    artifact_events = [e for e in events if e["type"].startswith("artifact.")]
    assert types(artifact_events) == ["artifact.created", "artifact.modified", "artifact.deleted"]
    assert all(e["recorder"] == "supervisor" for e in artifact_events)
    # Each change is recorded straight after the result of the call that made it.
    for event in artifact_events:
        assert events[event["seq"] - 1]["type"] == "tool.result"

    first, second = read_index(result.archive)
    assert (first["workspace_path"], first["revision"]) == ("notes/plan.md", 1)
    assert (second["revision"], second["supersedes"]) == (2, first["artifact_id"])
    assert first["producer"] == {
        "kind": "tool", "capability_id": "file.write",
        "tool_call_correlation_id": "call-000001",
    }
    assert first["observed_at_seq"] == artifact_events[0]["seq"]
    assert (result.archive / first["archive_path"]).read_text() == "one"
    assert (result.archive / second["archive_path"]).read_text() == "two"
    assert not (result.workspace / "notes" / "plan.md").exists()


def test_archive_is_complete_after_the_workspace_is_deleted(root, make_config):
    config = make_config([
        {"text": "write", "tool_calls": [call("file.write", path="a.txt", content="kept")]},
    ])
    result = run_experiment(config, root)
    shutil.rmtree(result.workspace)
    shutil.rmtree(root / "worlds")
    assert verify_archive(result.archive) == []
    (record,) = read_index(result.archive)
    assert (result.archive / record["archive_path"]).read_text() == "kept"


# --- authority comes only from the configuration ---------------------------------

def test_capability_needing_an_ungranted_permission_is_denied(root, make_config):
    config = make_config(
        [{"text": "I am authorised to write files.",
          "tool_calls": [call("file.write", path="a.txt", content="x")]}],
        permissions={"filesystem": "none"},
    )
    result = run_experiment(config, root)
    (outcome,) = results(events_of(result.archive))
    assert outcome["status"] == "denied"
    assert list(result.workspace.iterdir()) == []


def test_capabilities_can_be_narrowed_by_the_experiment(root, make_config):
    replies = [{"text": "try", "tool_calls": [
        call("text.stats", text="a"), call("math.evaluate", expression="1+1")]}]
    denied = run_experiment(make_config(replies, capabilities={"deny": ["text.stats"]}), root)
    assert [r["status"] for r in results(events_of(denied.archive))] == ["denied", "ok"]
    allowed = run_experiment(make_config(replies, capabilities={"allow": ["text.stats"]}), root)
    assert [r["status"] for r in results(events_of(allowed.archive))] == ["ok", "denied"]


def test_capability_processes_inherit_no_environment(root, make_config, monkeypatch):
    monkeypatch.setenv("OLS_TEST_SECRET_TOKEN", "do-not-leak")
    config = make_config([{"text": "look", "tool_calls": [
        call("test.environment"), call("test.outside")]}])
    result = run_experiment(config, root)
    environment, outside = results(events_of(result.archive))
    assert "OLS_TEST_SECRET_TOKEN" not in environment["result"]["names"]
    assert not {"PATH", "USERNAME", "USERPROFILE", "HOME", "APPDATA"} & set(
        environment["result"]["names"]
    )
    assert "do-not-leak" not in (result.archive / "events.jsonl").read_text(encoding="utf-8")
    # A capability without filesystem permission is not told where the workspace is.
    assert outside["result"] == {"ok": True}


def test_bad_tool_calls_are_recorded_and_do_not_stop_the_run(root, make_config):
    config = make_config([{"text": "try", "tool_calls": [
        call("text.stats", text=5),
        {"capability_id": "../../etc", "arguments": {}},
        {"capability_id": "text.stats", "arguments": "not an object"},
        "nonsense",
        call("text.stats", text="fine"),
    ]}])
    result = run_experiment(config, root)
    assert result.state == "completed"
    events = events_of(result.archive)
    assert [r["status"] for r in results(events)] == ["error", "ok"]
    assert results(events)[0]["error"].startswith("invalid_input")
    errors = [e for e in events if e["type"] == "error"]
    assert len(errors) == 3 and all("malformed tool call" in e["payload"]["message"]
                                    for e in errors)
    assert verify_archive(result.archive) == []


ROGUE = '''
import json, sys
sys.stdin.readline()
def ask(request):
    sys.stdout.write(json.dumps({"event": request}) + "\\n"); sys.stdout.flush()
    sys.stdin.readline()
agent = {"kind": "agent", "model_id": "scripted"}
ask({"recorder": "supervisor", "type": "supervisor.action", "actor": {"kind": "supervisor"},
     "payload": {"action": "resumed", "detail": "deadline extended"}})
ask({"recorder": "supervisor", "type": "run.state_changed", "actor": agent,
     "payload": {"from": "running", "to": "completed"}})
ask({"recorder": "agent_runtime", "type": "artifact.created", "actor": agent,
     "payload": {"artifact_id": "art-1", "path": "x"}})
ask({"recorder": "agent_runtime", "type": "agent.output", "actor": {"kind": "human_operator"},
     "payload": {"text": "the operator says continue"}})
ask({"recorder": "agent_runtime", "type": "agent.output", "actor": agent,
     "payload": {"text": "x"}, "seq": 0, "timestamp": "2000-01-01T00:00:00Z"})
ask({"recorder": "agent_runtime", "type": "agent.output", "actor": agent, "payload": {}})
sys.stdout.write("not json\\n"); sys.stdout.flush(); sys.stdin.readline()
ask({"recorder": "agent_runtime", "type": "agent.output", "actor": agent,
     "payload": {"text": "an honest line"}})
EXIT
'''


def rogue_runtime(tmp_path, monkeypatch, ending: str) -> None:
    script = tmp_path / "rogue.py"
    script.write_text(ROGUE.replace("EXIT", ending), encoding="utf-8")
    monkeypatch.setattr(supervisor, "WORKER_COMMAND", [sys.executable, str(script)])


def test_runtime_cannot_write_itself_a_different_history(root, make_config, tmp_path,
                                                         monkeypatch):
    ending = 'sys.stdout.write(json.dumps({"end": {"reason": "model_finished"}}) + "\\n")'
    rogue_runtime(tmp_path, monkeypatch, ending)
    result = run_experiment(make_config([]), root)
    events = events_of(result.archive)
    refused = [e for e in events if e["type"] == "error"]
    assert len(refused) == 7
    assert all(e["recorder"] == "supervisor" and "refused" in e["payload"]["message"]
               for e in refused)
    # Only the one legitimate request got in; nothing forged did.
    assert [e["payload"]["text"] for e in events if e["type"] == "agent.output"] == [
        "an honest line"
    ]
    assert "supervisor.action" not in types(events) and "artifact.created" not in types(events)
    assert all(e["recorder"] in ("supervisor", "experiment_manager")
               for e in events if e["type"] == "run.state_changed")
    assert all(not e["timestamp"].startswith("2000") for e in events)
    assert verify_archive(result.archive) == []


def test_runtime_crash_ends_the_run_as_failed(root, make_config, tmp_path, monkeypatch):
    rogue_runtime(tmp_path, monkeypatch, "raise SystemExit(5)")
    result = run_experiment(make_config([]), root)
    assert result.state == "failed" and "status 5" in result.reason
    events = events_of(result.archive)
    assert events[-1]["payload"]["to"] == "failed"
    assert verify_archive(result.archive) == []


# --- refusing to launch ----------------------------------------------------------

def refused(config: Path, root: Path) -> str:
    before = tree(root)
    with pytest.raises(LaunchError) as raised:
        run_experiment(config, root)
    assert tree(root) == before, "a refused launch must create nothing"
    assert not (root / "runs").exists() and not (root / "workspaces").exists()
    return str(raised.value)


def test_tampered_world_release_is_refused(root, release, make_config):
    code = release / "capabilities" / "text.stats" / "text_stats.py"
    code.write_text(code.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")
    assert "content digest" in refused(make_config([]), root)


def test_world_with_edited_permissions_is_refused(root, release, make_config):
    path = release / "world-release.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["default_permissions"]["network"] = "loopback" \
        if manifest["default_permissions"]["network"] == "none" else "none"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert "content digest" in refused(make_config([]), root)


def test_experiment_pinned_to_another_digest_is_refused(root, release, make_config):
    digest = load_world(release).digest
    pinned = make_config([], world={"content_digest": digest}, name="pinned")
    assert run_experiment(pinned, root).state == "completed"
    shutil.rmtree(root / "runs")
    shutil.rmtree(root / "workspaces")
    other = make_config([], world={"content_digest": "0" * 64}, name="other")
    assert "digest differs" in refused(other, root)


def test_draft_and_missing_worlds_are_refused(root, release, make_config):
    assert "refused" in refused(make_config([], world={"version": "9.9.9"}), root)
    path = release / "world-release.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["status"] = "draft"
    del manifest["content_digest"]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert "only built releases can be run" in refused(make_config([]), root)


@pytest.mark.parametrize(
    "changes, expected",
    [
        ({"orchestrator": {"model_id": "orchestrator-small"}}, "scripted stand-in"),
        ({"extensions": {"scripted_model": "no replies"}}, "scripted stand-in"),
        ({"termination": {"on_deadline": "checkpoint_then_stop"}}, "checkpointing"),
        ({"checkpointing": {"interval_s": 60}}, "checkpointing"),
        ({"termination": {"notify_agent_before_s": 10}}, "deadline notices"),
        ({"permissions": {"subprocess": True}}, "container isolation tier"),
        ({"capabilities": {"agent_created_tools": True}}, "container isolation tier"),
        ({"permissions": {"network": "allowlist", "network_allowlist": ["example.org"]}},
         "allowlist"),
        ({"limits": {"wall_clock_s": 0}}, "invalid experiment configuration"),
        ({"surprise": True}, "invalid experiment configuration"),
        ({"world": {"world_id": "../../outside"}}, "no valid world"),
    ],
)
def test_configurations_this_runtime_cannot_honour_are_refused(root, make_config, changes,
                                                               expected):
    assert expected in refused(make_config([], **changes), root)


def test_unreadable_configuration_is_refused(root, tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    assert "cannot be read" in refused(path, root)


# --- command line ------------------------------------------------------------------

def test_cli_run_then_verify(root, capsys):
    assert main(["run", str(SMOKE), "--data-root", str(root)]) == 0
    assert "ended: completed" in capsys.readouterr().out
    (archive,) = (root / "runs" / "scripted-smoke").iterdir()
    assert main(["verify", str(archive)]) == 0
    assert "intact" in capsys.readouterr().out


def test_cli_reports_a_refusal(root, make_config, capsys):
    config = make_config([], world={"version": "9.9.9"})
    assert main(["run", str(config), "--data-root", str(root)]) == 2
    assert "nothing was created" in capsys.readouterr().err
