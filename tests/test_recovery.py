"""Fault injection with real workers, OS locks and a fake local model server."""

import json
import os
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from conftest import call, events_of, tree
from open_lab_simulation import ownership, recovery, supervisor
from open_lab_simulation.archive import verify_archive
from open_lab_simulation.cli import main
from open_lab_simulation.ownership import OwnershipError, archive_lock, process_birth
from open_lab_simulation.recorder import Recorder
from open_lab_simulation.supervisor import (
    LaunchError,
    recover_interrupted,
    resume_experiment,
    run_experiment,
)
from open_lab_simulation.world import load_world
from test_model import FakeServer, point_world_at

POLICY = {"max_restarts": 2, "auto_resume": True}


def wait_until(predicate, timeout=10):
    stop = time.monotonic() + timeout
    while time.monotonic() < stop:
        try:
            value = predicate()
            if value:
                return value
        except (OSError, ValueError, StopIteration):
            pass
        time.sleep(0.01)
    raise AssertionError("timed out waiting for fault-injection boundary")


def stages(archive):
    return [recovery.event_extension(e).get("stage") for e in events_of(archive)]


def enable(config, **settings):
    document = json.loads(config.read_text())
    document["extensions"]["recovery"] = {**POLICY, **settings}
    config.write_text(json.dumps(document))
    return config


def interrupted_fixture(paused, release):
    # Private fixture for damaged-archive cases; the supervisor-death test below
    # independently exercises a real interruption instead of constructing one.
    recorder = Recorder(paused.archive / "events.jsonl", load_world(release).contracts,
                        "test-run", paused.run_id, append=True)
    recorder.record(recorder="supervisor", type="run.state_changed",
                    actor={"kind": "supervisor"},
                    payload={"from": "paused", "to": "running", "reason": "test fixture"})
    recorder.close()
    (paused.archive / "summary.json").unlink()


def crash_workers(monkeypatch, root, *, every=False):
    original = supervisor.ProcessTree
    threads, errors = [], []

    class CrashingTree(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if threads and not every:
                return
            index = len(threads) + 1

            def crash():
                try:
                    wait_until(lambda: sum(
                        recovery.event_extension(e).get("stage") == "model_step_reserved"
                        for archive in (root / "runs").glob("*/*")
                        for e in events_of(archive)
                    ) >= index)
                    self.process.kill()
                except Exception as error:
                    errors.append(error)

            thread = threading.Thread(target=crash)
            threads.append(thread)
            thread.start()

    monkeypatch.setattr(supervisor, "ProcessTree", CrashingTree)
    return threads, errors


def test_runtime_kill_recovers_under_same_supervisor_without_erasing_observations(
    root, make_config, monkeypatch,
):
    config = enable(make_config([
        {"delay_s": 0.4, "tool_calls": [call("workspace.write", path="result.txt", text="done")]},
    ], checkpointing={}, limits={"max_steps": 5}))
    threads, errors = crash_workers(monkeypatch, root)
    result = run_experiment(config, root)
    for thread in threads:
        thread.join()
    assert errors == []
    assert result.state == "completed" and result.notes == []
    assert (result.workspace / "result.txt").read_text() == "done"
    events = events_of(result.archive)
    assert recovery.restarts_used(events) == 1
    assert recovery.steps_used(events) >= 2  # killed attempt was not free
    assert "recovery_restart" in stages(result.archive)
    assert any(e["type"] == "recovery" for e in events)
    assert [e["seq"] for e in events] == list(range(len(events)))
    assert verify_archive(result.archive) == []


def test_repeated_runtime_crash_exhausts_one_shared_retry_budget(root, make_config, monkeypatch):
    config = enable(make_config([{"delay_s": 30}], checkpointing={}), max_restarts=1)
    threads, errors = crash_workers(monkeypatch, root, every=True)
    result = run_experiment(config, root)
    for thread in threads:
        thread.join()
    assert errors == [] and len(threads) == 2
    assert result.state == "failed"
    assert recovery.restarts_used(events_of(result.archive)) == 1
    assert "recovery_exhausted" in stages(result.archive)
    assert verify_archive(result.archive) == []
    with pytest.raises(LaunchError, match="retry budget exhausted"):
        resume_experiment(result.archive, root)


def test_deadline_and_interrupted_step_budget_are_not_reset_after_crash(
    root, make_config, monkeypatch,
):
    config = enable(make_config([{"delay_s": 30}], checkpointing={}, wall_clock_s=2))
    threads, errors = crash_workers(monkeypatch, root)
    started = time.monotonic()
    result = run_experiment(config, root)
    for thread in threads:
        thread.join()
    assert errors == []
    assert (result.state, result.reason) == ("terminated", "deadline")
    assert time.monotonic() - started < 5
    summary = json.loads((result.archive / "summary.json").read_text())
    assert 2 <= summary["elapsed_s"] < 3
    assert recovery.restarts_used(events_of(result.archive)) == 1
    assert verify_archive(result.archive) == []


def test_killed_attempt_uses_last_available_step_and_refuses_restart(
    root, make_config, monkeypatch,
):
    config = enable(make_config([{"delay_s": 30}], checkpointing={}, limits={"max_steps": 1}))
    threads, errors = crash_workers(monkeypatch, root)
    result = run_experiment(config, root)
    for thread in threads:
        thread.join()
    assert errors == [] and result.state == "failed"
    assert recovery.restarts_used(events_of(result.archive)) == 0
    assert recovery.steps_used(events_of(result.archive)) == 1
    assert "recovery_refused" in stages(result.archive)
    assert verify_archive(result.archive) == []


def test_live_os_writer_lock_cannot_be_displaced_even_if_marker_changes(tmp_path):
    archive = tmp_path / "run"
    archive.mkdir()
    with archive_lock(archive):
        marker = archive / ".writer.lock"
        original = marker.read_bytes()
        with pytest.raises(OwnershipError, match="live writer"):
            with archive_lock(archive, recover_stale=True):
                pytest.fail("second writer acquired the archive")
        assert marker.read_bytes() == original
    assert not marker.exists()


@pytest.mark.parametrize("kind", ["pid_reused", "dead"])
def test_stale_owner_verified_by_birth_and_evidence_retained(tmp_path, monkeypatch, kind):
    archive = tmp_path / "run"
    archive.mkdir()
    marker = archive / ".writer.lock"
    old = {"pid": 12345, "birth": "old-birth", "host": ownership.host_identity()}
    marker.write_text(json.dumps(old))
    original = process_birth
    monkeypatch.setattr(ownership, "process_birth", lambda pid:
                        ("different-birth" if kind == "pid_reused" else None)
                        if pid == 12345 else original(pid))
    with archive_lock(archive, recover_stale=True) as evidence:
        assert json.loads((archive / "segments" / evidence).read_text()) == old
        assert json.loads(marker.read_text())["pid"] == os.getpid()


@pytest.mark.parametrize("owner", ["bad", {"host": "another-host", "pid": 123, "birth": "x"}])
def test_unverifiable_owner_is_never_overwritten(tmp_path, owner):
    tmp_path.joinpath(".writer.lock").write_text(json.dumps(owner))
    before = tmp_path.joinpath(".writer.lock").read_bytes()
    with pytest.raises(OwnershipError, match="cannot be verified"):
        with archive_lock(tmp_path, recover_stale=True):
            pytest.fail("unknown owner was displaced")
    assert tmp_path.joinpath(".writer.lock").read_bytes() == before


def test_live_birth_identity_and_inspection_failure_both_refuse_takeover(tmp_path, monkeypatch):
    marker = tmp_path / ".writer.lock"
    marker.write_text(json.dumps({"pid": os.getpid(), "birth": process_birth(os.getpid()),
                                  "host": ownership.host_identity()}))
    before = marker.read_bytes()
    with pytest.raises(OwnershipError, match="live writer"):
        with archive_lock(tmp_path, recover_stale=True):
            pytest.fail("live owner displaced")

    def inaccessible(pid):
        raise OwnershipError("cannot inspect owner")

    monkeypatch.setattr(ownership, "process_birth", inaccessible)
    with pytest.raises(OwnershipError, match="cannot be verified"):
        with archive_lock(tmp_path, recover_stale=True):
            pytest.fail("unverifiable owner displaced")
    assert marker.read_bytes() == before


@pytest.mark.parametrize("damage", ["partial_log", "checkpoints", "accounting", "step_budget"])
def test_startup_recovery_refuses_untrustworthy_or_exhausted_state(
    root, release, make_config, damage,
):
    config = enable(make_config(
        [{"text": "first"}, {"text": "second"}], checkpointing={},
        limits={"max_steps": 1 if damage == "step_budget" else 5},
    ))
    paused = run_experiment(config, root, pause_after_steps=1)
    interrupted_fixture(paused, release)
    if damage == "partial_log":
        with (paused.archive / "events.jsonl").open("ab") as handle:
            handle.write(b'{"incomplete":')
    elif damage == "checkpoints":
        for folder in (paused.archive / "checkpoints").iterdir():
            (folder / "conversation.json").write_text("bad")
    elif damage == "accounting":
        (paused.archive / ".supervisor.json").write_text("bad")
    before = (paused.archive / "events.jsonl").read_bytes()
    results, refused = recover_interrupted(root)
    assert results == [] and len(refused) == 1
    expected = {"partial_log": "partial", "checkpoints": "no valid",
                "accounting": "Expecting value", "step_budget": "exhausted"}
    assert expected[damage] in refused[0][1]
    after = (paused.archive / "events.jsonl").read_bytes()
    if damage == "partial_log":
        assert after == before  # cannot safely append to a corrupt observation log
    else:
        assert after.startswith(before)
        assert stages(paused.archive)[-1] == "recovery_refused"
    assert not (paused.archive / "summary.json").exists()


def test_startup_scan_never_displaces_live_supervisor_or_replays_unopted_runs(root, make_config):
    config = enable(make_config([{"text": "one"}, {"text": "two"}], checkpointing={}))
    paused = run_experiment(config, root, pause_after_steps=1)
    summary = paused.archive / "summary.json"
    summary.unlink()
    before = (paused.archive / "events.jsonl").read_bytes()
    with archive_lock(paused.archive):
        results, refused = recover_interrupted(root)
        assert results == [] and "live writer" in refused[0][1]
    assert (paused.archive / "events.jsonl").read_bytes() == before
    document = json.loads((paused.archive / "experiment.json").read_text())
    document["extensions"]["recovery"]["auto_resume"] = False
    (paused.archive / "experiment.json").write_text(json.dumps(document))
    assert recover_interrupted(root) == ([], [])


def test_slow_health_probe_has_a_supervisor_timeout(monkeypatch):
    from open_lab_simulation import ollama

    release = threading.Event()

    def blocked(*args, **kwargs):
        release.wait(2)
        raise ollama.OllamaError("delayed failure")

    monkeypatch.setattr(ollama, "_request", blocked)
    monitor = recovery.HealthMonitor({"base_url": "http://127.0.0.1:1/v1"}, {
        "health_check_interval_s": 0.05, "health_check_timeout_s": 0.05,
    })
    try:
        assert monitor.poll() is None
        assert wait_until(monitor.poll, timeout=0.5) == "model health check timed out"
    finally:
        release.set()


@pytest.mark.parametrize("failure", ["write_error", "expired_lease"])
def test_accounting_failure_stops_worker_without_recovery(root, make_config, monkeypatch, failure):
    original = recovery.Accounting.attach

    def failed_attach(self, pid):
        original(self, pid)
        if failure == "write_error":
            self.error = OSError("injected accounting write failure")
        else:
            self.lease_deadline = time.monotonic() - 1

    monkeypatch.setattr(recovery.Accounting, "attach", failed_attach)
    result = run_experiment(enable(make_config([{"delay_s": 30}], checkpointing={})), root)
    assert (result.state, result.reason) == ("failed", "recovery accounting failed")
    assert recovery.restarts_used(events_of(result.archive)) == 0
    assert any(e["type"] == "supervisor.action" and e["payload"]["action"] == "terminated"
               for e in events_of(result.archive))
    assert verify_archive(result.archive) == []


def test_recovery_without_checkpoints_is_refused(root, make_config):
    with pytest.raises(LaunchError, match="recovery requires checkpointing"):
        run_experiment(enable(make_config([])), root)


@pytest.mark.parametrize("missing", ["process", "boot_identity"])
def test_linux_identity_distinguishes_absent_process_from_unverifiable_boot(monkeypatch, missing):
    def read(name):
        if name.endswith("/stat") and missing == "boot_identity":
            return "123 (worker) S " + "0 " * 18 + "42"
        raise FileNotFoundError

    with monkeypatch.context() as patch:
        patch.setattr(ownership.sys, "platform", "linux")
        patch.setattr(ownership, "Path", lambda name: SimpleNamespace(read_text=lambda: read(name)))
        if missing == "process":
            assert ownership.process_birth(123) is None
        else:
            with pytest.raises(OwnershipError, match="cannot inspect"):
                ownership.process_birth(123)


def test_missing_summary_after_completed_state_never_replays_run(root, make_config):
    result = run_experiment(enable(make_config([], checkpointing={})), root)
    (result.archive / "summary.json").unlink()
    results, refused = recover_interrupted(root)
    assert results == [] and "already records a stopped run" in refused[0][1]
    assert "recovery_restart" not in stages(result.archive)


@pytest.mark.skipif(sys.platform != "win32", reason="native supervisor-death Job Object check")
def test_startup_scan_recovers_killed_supervisor_and_excludes_downtime(root, make_config):
    config = enable(make_config([
        {"delay_s": 0.5, "tool_calls": [call("workspace.write", path="done.txt", text="done")]},
    ], checkpointing={}, wall_clock_s=15))
    process = subprocess.Popen(
        [sys.executable, "-m", "open_lab_simulation.cli", "run", str(config),
         "--data-root", str(root)], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    try:
        archive = wait_until(lambda: next((root / "runs/test-run").iterdir()))
        wait_until(lambda: "model_step_reserved" in stages(archive))
        runtime = wait_until(lambda: json.loads(
            (archive / ".supervisor.json").read_text()
        )["runtime"])
        owner = json.loads((archive / ".writer.lock").read_text())
        assert process_birth(owner["pid"]) == owner["birth"]
        assert process_birth(runtime["pid"]) == runtime["birth"]
        # A Windows venv redirector can have a different PID from Python itself.
        # Inject the fault into the archive's verified supervisor, not its launcher.
        killed = subprocess.run(["taskkill", "/PID", str(owner["pid"]), "/F"],
                                capture_output=True, check=False)
        assert killed.returncode == 0
        process.wait(timeout=10)
        wait_until(lambda: process_birth(owner["pid"]) != owner["birth"])
        wait_until(lambda: process_birth(runtime["pid"]) != runtime["birth"])
        old_log = (archive / "events.jsonl").read_bytes()
        assert (archive / ".writer.lock").exists() and not (archive / "summary.json").exists()
        time.sleep(2)
        results, refused = recover_interrupted(root)
        assert refused == [] and len(results) == 1
        result = results[0]
        assert result.state == "completed" and result.run_id == archive.name
        assert (archive / "events.jsonl").read_bytes().startswith(old_log)
        assert (result.workspace / "done.txt").read_text() == "done"
        assert list((archive / "segments").glob("writer-*.json"))
        assert recovery.restarts_used(events_of(archive)) == 1
        assert verify_archive(archive) == []
        assert main(["recover", "--data-root", str(root)]) == 0  # completed run not replayed
        from datetime import datetime
        summary = json.loads((archive / "summary.json").read_text())
        wall = (datetime.fromisoformat(summary["ended_at"])
                - datetime.fromisoformat(summary["started_at"])).total_seconds()
        assert wall - summary["elapsed_s"] > 0.7
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
        process.stderr.close()


@pytest.mark.parametrize("change", [
    {"max_restarts": -1}, {"max_restarts": True}, {"auto_resume": "yes"},
    {"health_check_interval_s": 0}, {"surprise": 1},
])
def test_invalid_recovery_policy_refused_before_creating_run(root, make_config, change):
    config = enable(make_config([], checkpointing={}), **change)
    before = tree(root)
    with pytest.raises(LaunchError, match="recovery"):
        run_experiment(config, root)
    assert tree(root) == before


def test_model_health_detects_missing_identity_during_inflight_request(root, release, make_config):
    fake = FakeServer()
    threads, errors = [], []
    try:
        point_world_at(release, fake.base + "/v1")
        config = enable(make_config([], checkpointing={},
            orchestrator={"provider_id": "ollama-local", "model_id": "orchestrator-small"},
            permissions={"network": "loopback", "models": ["orchestrator-small"]}),
            health_check_interval_s=0.05, health_check_timeout_s=0.1)
        fake.replies = [("delay", 3)]

        def remove_model():
            try:
                archive = wait_until(lambda: next((root / "runs/test-run").iterdir()))
                wait_until(lambda: any(
                    recovery.event_extension(e).get("status") == "healthy"
                    for e in events_of(archive)
                ))
                fake.models = []
            except Exception as error:
                errors.append(error)

        thread = threading.Thread(target=remove_model)
        threads.append(thread)
        thread.start()
        result = run_experiment(config, root)
        thread.join()
        assert errors == []
        assert result.state == "failed" and result.reason == "model health check failed"
        assert "model_health" in stages(result.archive)
        assert "recovery_refused" in stages(result.archive)
        assert verify_archive(result.archive) == []
    finally:
        fake.close()
