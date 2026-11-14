"""Native bounded resource faults and truthful resource samples; no real model."""

import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from conftest import FIXTURE_WORLD, TEST_CAPABILITIES, add_test_capabilities, call, events_of, tree
from open_lab_simulation import resources
from open_lab_simulation.archive import verify_archive
from open_lab_simulation.proctree import ProcessTree
from open_lab_simulation.supervisor import LaunchError, resume_experiment, run_experiment


@pytest.fixture
def resource_root(monkeypatch, tmp_path):
    # Fixed test-only code, not a shipped capability or agent-supplied program.
    tools = {
        "test.fill": ({"filesystem": "workspace_write", "network": "none"}, 20, '''
import os
from pathlib import Path
def run(arguments):
    (Path(os.environ["OPEN_LAB_WORKSPACE"]) / "full.txt").write_bytes(b"x" * 1024**2)
    return {"ok": True}
'''),
        "test.allocate": ({"filesystem": "none", "network": "none"}, 20, '''
import time
def run(arguments):
    blocks = []
    try:
        while True:
            blocks.append(bytearray(1024*1024))
    except MemoryError:
        time.sleep(10)
    return {"ok": True}
'''),
        "test.cpu": ({"filesystem": "none", "network": "none"}, 20, '''
import time
def run(arguments):
    until = time.monotonic() + arguments.get("seconds", 2)
    value = 1
    while time.monotonic() < until:
        value = (value * 7 + 1) % 1000007
    return {"ok": True}
'''),
        "test.grow": ({"filesystem": "workspace_write", "network": "none"}, 20, '''
import os, time
from pathlib import Path
def run(arguments):
    with (Path(os.environ["OPEN_LAB_WORKSPACE"]) / "growing.txt").open("wb") as f:
        for _ in range(1000):
            f.write(b"x" * 65536)
            f.flush()
            time.sleep(0.01)
    return {"ok": True}
'''),
    }
    for name, spec in tools.items():
        monkeypatch.setitem(TEST_CAPABILITIES, name, spec)
    root = tmp_path / "data"
    release = root / "worlds/minimal/0.6.0"
    shutil.copytree(FIXTURE_WORLD, release)
    add_test_capabilities(release)
    return root


def actions(result):
    return [e["payload"] for e in events_of(result.archive) if e["type"] == "supervisor.action"]


@pytest.mark.skipif(sys.platform != "win32", reason="Windows Job Object native enforcement")
@pytest.mark.parametrize("limit, reply", [
    ({"ram_mb": 160}, call("test.allocate")),
    ({"max_processes": 2}, call("text.stats", text="must not run")),
])
def test_native_job_limit_terminates_tree_and_records_reason(
    resource_root, make_config, limit, reply,
):
    result = run_experiment(make_config(
        [{"tool_calls": [reply]}], wall_clock_s=12, limits={"resources": limit},
    ), resource_root)
    name = next(iter(limit))
    assert (result.state, result.reason) == ("terminated", f"resource_limit:{name}")
    assert result.notes == []
    assert any(a["action"] == "terminated" and a["extensions"]["limit"] == name
               for a in actions(result))
    assert verify_archive(result.archive) == []


@pytest.mark.skipif(sys.platform != "win32", reason="Windows CPU rate hard cap")
def test_native_cpu_hard_cap_and_applied_limit_event(resource_root, make_config):
    result = run_experiment(make_config(
        [{"tool_calls": [call("test.cpu", seconds=2)]}], wall_clock_s=15,
        limits={"resources": {"cpu_cores": 0.25}},
        extensions={"resource_sampling": {"interval_s": 0.2}},
    ), resource_root)
    assert result.state == "completed", result.notes
    assert any(a["extensions"]["limit"] == "cpu_cores" for a in actions(result))
    assert verify_archive(result.archive) == []
    # Measure the kernel cap independently of the supervisor's event rendering.
    job = ProcessTree([getattr(sys, "_base_executable", sys.executable), "-c",
                       "while True: pass"], resources={"cpu_cores": 0.2},
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        job.sample()
        time.sleep(2)
        measured = job.sample()["cpu_percent"]
        assert 0 < measured < 45, measured  # 0.2 core, with scheduling/startup tolerance
    finally:
        job.close()


def test_quota_denies_subsequent_writes_but_allows_read_only_tools(resource_root, make_config):
    result = run_experiment(make_config([
        {"tool_calls": [call("test.fill")]},
        {"tool_calls": [call("workspace.write", path="denied.txt", text="no")]},
        {"tool_calls": [call("workspace.list")]},
        {"tool_calls": [{"capability_id": "capabilities.demonstrate",
                         "arguments": {"id": "workspace.write"}}]},
    ], limits={"resources": {"disk_mb": 1}}), resource_root)
    assert result.state == "completed"
    outcomes = [e["payload"] for e in events_of(result.archive) if e["type"] == "tool.result"]
    assert [o["status"] for o in outcomes] == ["ok", "denied", "ok", "denied"]
    assert "quota" in outcomes[1]["error"]
    assert not (result.workspace / "denied.txt").exists()
    assert any(a["action"] == "permission_denied" and a["extensions"]["limit"] == "disk_mb"
               for a in actions(result))
    assert verify_archive(result.archive) == []


def test_quota_kills_an_inflight_writer_and_does_not_retry(resource_root, make_config):
    result = run_experiment(make_config([
        {"tool_calls": [call("test.grow")]},
    ], limits={"resources": {"disk_mb": 1}}, checkpointing={},
        extensions={"resource_sampling": {"interval_s": 60},
                    "recovery": {"max_restarts": 2}}), resource_root)
    assert (result.state, result.reason) == ("terminated", "resource_limit:disk_mb")
    path = result.workspace / "growing.txt"
    size = path.stat().st_size
    assert size > 1024**2
    time.sleep(0.15)
    assert path.stat().st_size == size
    assert not any(a["extensions"].get("stage") == "recovery_restart" for a in actions(result))
    assert verify_archive(result.archive) == []


def test_resource_caps_reinstalled_after_checkpoint_resume(resource_root, make_config):
    config = make_config([
        {"text": "first"},
        {"tool_calls": [call("test.fill")]},
        {"tool_calls": [call("workspace.write", path="denied.txt", text="no")]},
    ], checkpointing={}, limits={"resources": {"disk_mb": 1}})
    paused = run_experiment(config, resource_root, pause_after_steps=1)
    result = resume_experiment(paused.archive, resource_root)
    assert result.state == "completed"
    installed = [a for a in actions(result) if a.get("extensions", {}).get("stage") ==
                 "resource_limit_installed"]
    assert len(installed) == 2
    assert not (result.workspace / "denied.txt").exists()
    assert verify_archive(result.archive) == []


@pytest.mark.parametrize("limits", [{"vram_mb": 0}, {"wall_clock_s": 1}, {"cpu_cores": 100000}])
def test_unenforceable_limits_refused_before_creating_archive(root, make_config, limits):
    before = tree(root)
    with pytest.raises(LaunchError, match="enforced|represented"):
        run_experiment(make_config([], limits={"resources": limits}), root)
    assert tree(root) == before


@pytest.mark.parametrize("limit", [{"max_processes": 2**32}, {"ram_mb": 2**64}])
def test_native_integer_overflow_limits_refused(root, make_config, limit):
    with pytest.raises(LaunchError, match="represented"):
        run_experiment(make_config([], limits={"resources": limit}), root)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows resource setup failure")
def test_cpu_backend_failure_never_runs_worker(root, make_config, monkeypatch):
    from open_lab_simulation import proctree
    original = proctree._kernel32.SetInformationJobObject
    monkeypatch.setattr(proctree._kernel32, "SetInformationJobObject", lambda job, kind, *args:
                        0 if kind == 15 else original(job, kind, *args))
    result = run_experiment(make_config([], limits={"resources": {"cpu_cores": 1}}), root)
    assert result.state == "failed" and result.notes
    assert not any(e["type"] == "model.call" for e in events_of(result.archive))
    assert any(a["action"] == "terminated" for a in actions(result))
    assert verify_archive(result.archive) == []


def test_model_token_limit_not_silently_accepted(root, make_config):
    with pytest.raises(LaunchError, match="max_model_tokens"):
        run_experiment(make_config([], limits={"max_model_tokens": 1}), root)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows suspended startup ordering")
def test_process_is_suspended_until_assigned_to_job(monkeypatch):
    from open_lab_simulation import proctree
    events = []
    popen = proctree.subprocess.Popen
    assign = proctree._kernel32.AssignProcessToJobObject
    resume = proctree._resume_suspended

    def suspended(*args, **kwargs):
        assert kwargs["creationflags"] & 4
        events.append("created_suspended")
        return popen(*args, **kwargs)

    def assigned(*args):
        result = assign(*args)
        assert result
        events.append("assigned")
        return result

    def resumed(pid):
        assert events == ["created_suspended", "assigned"]
        events.append("resumed")
        return resume(pid)

    monkeypatch.setattr(proctree.subprocess, "Popen", suspended)
    monkeypatch.setattr(proctree._kernel32, "AssignProcessToJobObject", assigned)
    monkeypatch.setattr(proctree, "_resume_suspended", resumed)
    job = ProcessTree([sys.executable, "-c", "pass"], stdout=subprocess.DEVNULL)
    try:
        assert job.process.wait(timeout=10) == 0
        assert events == ["created_suspended", "assigned", "resumed"]
    finally:
        job.close()


@pytest.mark.parametrize("failure", ["missing", "timeout"])
def test_gpu_probe_failure_is_unavailable(monkeypatch, failure):
    probe = resources.GpuProbe()
    monkeypatch.setattr(resources.shutil, "which",
                        lambda name: None if failure == "missing" else name)

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("nvidia-smi", 2)

    monkeypatch.setattr(resources.subprocess, "run", timeout)
    probe.pending = True
    probe._read()
    assert probe.value is None and not probe.pending


def test_unreadable_workspace_stops_before_unchecked_work(root, make_config, monkeypatch):
    def unreadable(workspace):
        raise OSError("injected quota scan failure")

    monkeypatch.setattr(resources, "workspace_bytes", unreadable)
    result = run_experiment(make_config([{"delay_s": 10}],
                           limits={"resources": {"disk_mb": 1}}), root)
    assert (result.state, result.reason) == ("terminated", "resource_limit:monitoring")
    assert any(a["action"] == "terminated" for a in actions(result))
    assert verify_archive(result.archive) == []


def test_non_windows_job_limits_refused(monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(resources.sys, "platform", "linux")
        with pytest.raises(ValueError, match="Windows"):
            resources.validate({"limits": {"resources": {"ram_mb": 100}}})


@pytest.mark.parametrize("value", [0, True, float("inf"), "fast"])
def test_invalid_sampling_settings_refused(root, make_config, value):
    with pytest.raises(LaunchError, match="resource_sampling"):
        run_experiment(make_config([], extensions={"resource_sampling": {"interval_s": value}}),
                       root)


def test_gpu_zero_distinct_from_unavailable_and_aggregate():
    assert resources.parse_gpu("0, 4096\n") == {"used_mb": 0, "available_mb": 4096}
    assert resources.parse_gpu("1, 8\n2, 8") == {"used_mb": 3, "available_mb": 13}
    for invalid in ("", "N/A, 100", "-1, 8", "9, 8", "nan, 8"):
        with pytest.raises(ValueError):
            resources.parse_gpu(invalid)


@pytest.mark.parametrize("gpu", [None, {"used_mb": 0, "available_mb": 4096, "age_s": 0}])
def test_sample_availability_does_not_invent_zero(tmp_path, monkeypatch, gpu):
    captured = []
    recorder = SimpleNamespace(record=lambda **event: captured.append(event))
    world = SimpleNamespace(capabilities={})
    monitor = resources.Controller({"limits": {}}, tmp_path, world, recorder)
    monkeypatch.setattr(resources, "GPU", SimpleNamespace(poll=lambda: gpu))
    job = SimpleNamespace(violations=lambda: [], sample=lambda: {"process_count": 0})
    assert monitor.tick(job, {}) is None
    payload = captured[-1]["payload"]
    assert payload["process_count"] == payload["workspace_disk_mb"] == 0
    assert "ram_mb" not in payload and "ram_mb" in payload["extensions"]["unavailable"]
    if gpu is None:
        assert "vram_mb" not in payload and "vram_mb" in payload["extensions"]["unavailable"]
    else:
        assert payload["vram_mb"] == 0
        assert payload["extensions"]["vram_available_mb"] == 4096


def test_periodic_samples_validate_and_runtime_cannot_emit_them(root, make_config):
    result = run_experiment(make_config(
        [{"delay_s": 0.35}], extensions={"resource_sampling": {"interval_s": 0.1}},
    ), root)
    samples = [e for e in events_of(result.archive) if e["type"] == "resource.sample"]
    assert len(samples) >= 3
    assert all(e["recorder"] == "supervisor" for e in samples)
    from open_lab_simulation.supervisor import _RUNTIME_EVENTS
    assert "resource.sample" not in _RUNTIME_EVENTS
    assert verify_archive(result.archive) == []
