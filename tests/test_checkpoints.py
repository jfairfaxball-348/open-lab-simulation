"""Pause/resume acceptance, corruption fallback, budgets and append-only records."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import zipfile
from datetime import datetime

import pytest

from conftest import call, events_of, tree
from open_lab_simulation import supervisor
from open_lab_simulation.archive import verify_archive
from open_lab_simulation.checkpoints import (
    CheckpointError,
    CheckpointTimeout,
    _remove_owned_tree,
    checkpoint_events,
    restore_workspace,
    write_checkpoint,
)
from open_lab_simulation.cli import main
from open_lab_simulation.recorder import Recorder, RecordError
from open_lab_simulation.supervisor import LaunchError, resume_experiment, run_experiment
from open_lab_simulation.world import load_world


def replies(delay=0):
    return [
        {"delay_s": delay, "tool_calls": [call("memory.write", key="idea", value="first")]},
        {"tool_calls": [call("workspace.write", path="NOTES.md", text="working note")]},
        {"tool_calls": [call("memory.write", key="idea", value="revised")]},
        {"tool_calls": [call("workspace.write", path="final.txt", text="done")]},
    ]


def content(workspace):
    return {
        p.relative_to(workspace).as_posix(): p.read_bytes()
        for p in workspace.rglob("*")
        if p.is_file() and not p.name.startswith(".memory.sqlite")
    }


def memory(workspace):
    with sqlite3.connect(workspace / ".memory.sqlite") as connection:
        return connection.execute("SELECT key,value,revision FROM notes ORDER BY key").fetchall()


def test_stop_resume_matches_uninterrupted_workspace_memory_and_log(root, make_config):
    config = make_config(replies(), checkpointing={"interval_s": 1, "keep_last": 8})
    reference = run_experiment(config, root)
    paused = run_experiment(config, root, pause_after_steps=2)
    assert paused.state == "paused" and verify_archive(paused.archive) == []
    original_log = (paused.archive / "events.jsonl").read_bytes()
    (paused.archive / "runtime.stderr.log").write_bytes(b"first segment diagnostics\n")
    checkpoints = checkpoint_events(events_of(paused.archive))
    metadata = json.loads(
        (
            paused.archive
            / "checkpoints"
            / checkpoints[-1]["payload"]["checkpoint_id"]
            / "checkpoint.json"
        ).read_text()
    )
    assert metadata["progress"]["step_count"] == 2
    assert metadata["reason"] == "operator_request"
    assert {c["name"] for c in metadata["components"]} == {
        "conversation",
        "runtime_state",
        "memory",
        "workspace",
    }
    resumed = resume_experiment(paused.archive, root)
    assert resumed.state == "completed" and resumed.run_id == paused.run_id
    assert (resumed.archive / "events.jsonl").read_bytes().startswith(original_log)
    assert (
        (resumed.archive / "runtime.stderr.log")
        .read_bytes()
        .startswith(b"first segment diagnostics\n")
    )
    assert content(resumed.workspace) == content(reference.workspace)
    assert memory(resumed.workspace) == memory(reference.workspace) == [("idea", "revised", 2)]
    events = events_of(resumed.archive)
    assert [e["seq"] for e in events] == list(range(len(events)))
    ids = [e["correlation_id"] for e in events if e["type"] == "tool.call"]
    assert len(ids) == len(set(ids))
    assert (
        len(
            [
                e
                for e in events
                if e["type"] == "environment.message" and e["payload"]["kind"] == "preamble"
            ]
        )
        == 1
    )
    assert len([e for e in events if e["type"] == "memory.written"]) == 2
    assert verify_archive(resumed.archive) == []
    assert all(
        a["monotonic_ms"] <= b["monotonic_ms"] for a, b in zip(events, events[1:], strict=False)
    )


def test_corrupt_latest_checkpoint_falls_back_without_erasing_observations(root, make_config):
    config = make_config(
        replies(1.05), checkpointing={"interval_s": 1, "keep_last": 8}, limits={"max_steps": 10}
    )
    paused = run_experiment(config, root, pause_after_steps=2)
    checkpoints = checkpoint_events(events_of(paused.archive))
    assert len(checkpoints) >= 3, "initial, interval after delayed step, operator pause"
    latest = checkpoints[-1]["payload"]["checkpoint_id"]
    prior = checkpoints[-2]["payload"]["checkpoint_id"]
    (paused.archive / "checkpoints" / latest / "conversation.json").write_text("corrupted")
    old_log = (paused.archive / "events.jsonl").read_bytes()
    assert verify_archive(paused.archive), "corruption must be visible in archive verification"
    resumed = resume_experiment(paused.archive, root)
    assert resumed.state == "completed" and memory(resumed.workspace) == [("idea", "revised", 2)]
    assert content(resumed.workspace) == {"NOTES.md": b"working note", "final.txt": b"done"}
    recovery = next(e for e in events_of(resumed.archive) if e["type"] == "recovery")
    assert recovery["payload"]["checkpoint_id"] == prior
    assert recovery["payload"]["extensions"]["rejected_checkpoints"] == [latest]
    assert (resumed.archive / "events.jsonl").read_bytes().startswith(old_log)
    assert verify_archive(resumed.archive, checkpoints=False) == []
    assert verify_archive(resumed.archive), "known corrupt snapshot remains forensic evidence"


def test_elapsed_time_excludes_pause_downtime(root, make_config):
    config = make_config(replies(), checkpointing={})
    paused = run_experiment(config, root, pause_after_steps=1)
    first = json.loads((paused.archive / "summary.json").read_text())
    time.sleep(1.2)
    resumed = resume_experiment(paused.archive, root)
    final = json.loads((resumed.archive / "summary.json").read_text())
    wall = (
        datetime.fromisoformat(final["ended_at"]) - datetime.fromisoformat(first["started_at"])
    ).total_seconds()
    assert final["elapsed_s"] > first["elapsed_s"]
    assert wall - final["elapsed_s"] > 1.0


def test_retention_and_normal_shutdown_checkpoint_metadata_validate(root, make_config, release):
    result = run_experiment(
        make_config(replies(1.05), checkpointing={"interval_s": 1, "keep_last": 1}), root
    )
    folders = list((result.archive / "checkpoints").iterdir())
    assert len(folders) == 1
    metadata = json.loads((folders[0] / "checkpoint.json").read_text())
    assert load_world(release).contracts.validate(metadata, "checkpoint-metadata") == []
    assert metadata["reason"] == "pre_shutdown"
    assert metadata["progress"]["step_count"] == 4
    assert verify_archive(result.archive) == []


def test_original_step_budget_is_not_reset_on_resume(root, make_config):
    config = make_config(replies(), checkpointing={}, limits={"max_steps": 2})
    paused = run_experiment(config, root, pause_after_steps=1)
    resumed = resume_experiment(paused.archive, root)
    assert resumed.state == "terminated" and resumed.reason == "max_steps"
    assert len([e for e in events_of(resumed.archive) if e["type"] == "model.call"]) == 2
    with pytest.raises(LaunchError, match="exhausted"):
        resume_experiment(resumed.archive, root)


def test_deadline_budget_survives_resume_and_still_kills_busy_runtime(root, make_config):
    config = make_config([{"delay_s": 0.4}, {"delay_s": 10}], checkpointing={}, wall_clock_s=2)
    paused = run_experiment(config, root, pause_after_steps=1)
    resumed = resume_experiment(paused.archive, root)
    assert resumed.state == "terminated" and resumed.reason == "deadline"
    with pytest.raises(LaunchError, match="exhausted"):
        resume_experiment(resumed.archive, root)
    assert verify_archive(resumed.archive) == []


def test_live_lock_tampered_log_and_missing_all_checkpoints_are_refused(root, make_config):
    paused = run_experiment(make_config(replies(), checkpointing={}), root, pause_after_steps=1)
    lock = paused.archive / ".writer.lock"
    lock.write_text("existing owner")
    with pytest.raises(LaunchError, match="writer lock"):
        resume_experiment(paused.archive, root)
    assert lock.read_text() == "existing owner"
    lock.unlink()
    for folder in (paused.archive / "checkpoints").iterdir():
        (folder / "conversation.json").write_text("bad")
    before = tree(root)
    with pytest.raises(LaunchError, match="no valid"):
        resume_experiment(paused.archive, root)
    assert tree(root) == before
    with (paused.archive / "events.jsonl").open("a") as handle:
        handle.write("partial")
    with pytest.raises(LaunchError, match="cannot be resumed"):
        resume_experiment(paused.archive, root)


def test_completed_runs_and_unconfigured_operator_pause_are_refused(root, make_config):
    config = make_config([])
    with pytest.raises(LaunchError, match="checkpointing"):
        run_experiment(config, root, pause_after_steps=1)
    completed = run_experiment(config, root)
    with pytest.raises(LaunchError, match="incomplete"):
        resume_experiment(completed.archive, root)


def test_restore_preserves_post_pause_operator_edits(root, make_config):
    paused = run_experiment(make_config(replies(), checkpointing={}), root, pause_after_steps=1)
    (paused.workspace / "operator.txt").write_text("keep this")
    resumed = resume_experiment(paused.archive, root)
    assert not (resumed.workspace / "operator.txt").exists()
    backups = list(resumed.workspace.parent.glob(resumed.run_id + ".before-resume-*"))
    assert len(backups) == 1 and (backups[0] / "operator.txt").read_text() == "keep this"


@pytest.mark.parametrize("name", ["../escape", "/absolute", "C:/outside", "a\\b"])
def test_workspace_snapshot_path_traversal_is_refused(tmp_path, name):
    archive = tmp_path / "archive"
    folder = archive / "checkpoints/ckpt-00000001"
    folder.mkdir(parents=True)
    with zipfile.ZipFile(folder / "workspace.zip", "w") as bundle:
        bundle.writestr(name.replace("\\", "/"), "bad")
    if "\\" in name:
        # Windows' ZipInfo constructor normalises separators when writing.
        # Put the unsafe original spelling into both stored filename headers.
        path = folder / "workspace.zip"
        path.write_bytes(path.read_bytes().replace(b"a/b", b"a\\b"))
    workspace = tmp_path / "workspaces/run-1"
    workspace.mkdir(parents=True)
    (workspace / "existing").write_text("keep")
    with pytest.raises(CheckpointError, match="restore failed"):
        restore_workspace(
            archive, {"checkpoint_id": folder.name}, {"memory_present": False}, workspace
        )
    assert content(workspace) == {"existing": b"keep"}
    assert not (tmp_path / "escape").exists()


def test_recorder_refuses_partial_append_and_wrong_run(root, make_config, release):
    paused = run_experiment(make_config([], checkpointing={}), root)
    contracts = load_world(release).contracts
    with pytest.raises(RecordError, match="identity"):
        Recorder(paused.archive / "events.jsonl", contracts, "other", paused.run_id, append=True)
    with (paused.archive / "events.jsonl").open("ab") as handle:
        handle.write(b"{")
    with pytest.raises(RecordError, match="partial"):
        Recorder(paused.archive / "events.jsonl", contracts, "test-run", paused.run_id, append=True)


def test_cli_pause_resume_and_verify(root, make_config, capsys):
    config = make_config(replies(), checkpointing={})
    assert main(["run", str(config), "--data-root", str(root), "--pause-after-steps", "2"]) == 0
    assert "paused" in capsys.readouterr().out
    (archive,) = (root / "runs/test-run").iterdir()
    assert main(["resume", str(archive), "--data-root", str(root)]) == 0
    assert "completed" in capsys.readouterr().out
    assert main(["verify", str(archive)]) == 0


def test_snapshot_timeout_leaves_no_published_snapshot(root, release, tmp_path):
    world = load_world(release)
    archive = tmp_path / "archive"
    archive.mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    recorder = Recorder(archive / "events.jsonl", world.contracts, "exp-1", "run-1")
    recorder.record(
        recorder="supervisor",
        type="run.state_changed",
        actor={"kind": "supervisor"},
        payload={"to": "running"},
    )
    with pytest.raises(CheckpointTimeout):
        write_checkpoint(
            archive,
            workspace,
            recorder,
            world,
            "scripted",
            {},
            0,
            "interval",
            deadline=time.monotonic() - 1,
        )
    recorder.close()
    assert list((archive / "checkpoints").iterdir()) == []
    assert checkpoint_events(events_of(archive)) == []


def test_supervisor_enforces_deadline_if_snapshot_writer_runs_out_of_time(
    root,
    make_config,
    monkeypatch,
):
    def timeout(*args, **kwargs):
        assert kwargs["deadline"] > 0
        raise CheckpointTimeout("test deadline during snapshot")

    monkeypatch.setattr(supervisor, "write_checkpoint", timeout)
    result = run_experiment(make_config(replies(), checkpointing={}), root)
    assert result.state == "terminated" and result.reason == "deadline"
    events = events_of(result.archive)
    assert any(
        e["type"] == "supervisor.action" and e["payload"]["action"] == "terminated" for e in events
    )
    assert "model.call" not in [e["type"] for e in events]
    assert verify_archive(result.archive) == []


def test_operator_interrupt_requests_pause_at_next_complete_step(root, make_config, monkeypatch):
    original_get = supervisor.queue.Queue.get
    interrupted = False

    def interrupted_get(queue, *args, **kwargs):
        nonlocal interrupted
        if not interrupted and threading.current_thread() is threading.main_thread():
            interrupted = True
            raise KeyboardInterrupt
        return original_get(queue, *args, **kwargs)

    monkeypatch.setattr(supervisor.queue.Queue, "get", interrupted_get)
    paused = run_experiment(make_config(replies(), checkpointing={}), root)
    assert interrupted and paused.state == "paused"
    assert verify_archive(paused.archive) == []
    resumed = resume_experiment(paused.archive, root)
    assert resumed.state == "completed" and memory(resumed.workspace) == [("idea", "revised", 2)]


def test_checkpoint_cleanup_cannot_remove_its_root_or_a_foreign_directory(tmp_path):
    boundary = tmp_path / "staging"
    foreign = tmp_path / "foreign"
    boundary.mkdir()
    foreign.mkdir()
    (foreign / "keep.txt").write_text("keep")
    for target in (boundary, foreign):
        with pytest.raises(CheckpointError, match="refusing to remove"):
            _remove_owned_tree(target, boundary)
    assert boundary.is_dir() and (foreign / "keep.txt").read_text() == "keep"
