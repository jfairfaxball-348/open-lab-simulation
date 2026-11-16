"""S015: the Habitat world on the live container tier.

The inhabitant's own code runs through python.run, which the world offers only
at the container tier. These tests run on the actual backend (rootless Podman in
WSL 2) and are skipped when it is absent, unless OPEN_LAB_REQUIRE_CONTAINER=1,
in which case they fail. Milestone evidence is a run with that variable set.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from conftest import call, events_of
from open_lab_simulation.archive import verify_archive
from open_lab_simulation.supervisor import LaunchError, run_experiment
from test_container import backend  # noqa: F401 - live backend fixture

REPO = Path(__file__).resolve().parent.parent
HABITAT = REPO / "tests" / "fixtures" / "worlds" / "habitat" / "0.1.0"
DEMO = REPO / "experiments" / "habitat" / "s015-demo.json"


@pytest.fixture
def habitat_root(tmp_path):
    root = tmp_path / "data"
    shutil.copytree(HABITAT, root / "worlds" / "habitat" / "0.1.0")
    return root


def results(archive):
    return [e["payload"] for e in events_of(archive) if e["type"] == "tool.result"]


def habitat_config(tmp_path, replies, *, subprocess=True, name="habitat-check"):
    config = json.loads(DEMO.read_text(encoding="utf-8"))
    config["experiment_id"] = name
    config["permissions"]["subprocess"] = subprocess
    config["limits"]["wall_clock_s"] = 300
    config["extensions"]["scripted_model"]["replies"] = replies
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_the_demonstration_writes_runs_fixes_and_reuses_a_program(
        habitat_root, tmp_path, backend):  # noqa: F811
    path = tmp_path / "demo.json"
    shutil.copyfile(DEMO, path)
    result = run_experiment(path, habitat_root)
    assert result.state == "completed", result.reason
    found = results(result.archive)
    assert all(r["status"] == "ok" for r in found), [r.get("error") for r in found]
    by_step = [r["result"] for r in found]
    search, first_run, fixed_run = by_step[0], by_step[3], by_step[6]
    assert "python.run" in [m["id"] for m in search["matches"]]
    assert first_run["exit_code"] == 1 and "NameError" in first_run["stderr"]
    assert fixed_run["exit_code"] == 0
    assert fixed_run["stdout"] == "[2, 3, 5, 7, 11, 13, 17, 19, 23, 29]\n"
    assert by_step[9]["stdout"] == "25\n"  # report.py imported primes.py
    assert by_step[10]["text"] == "25 primes up to 100; largest 97\n"
    assert by_step[13]["entries"][0]["key"] == "skill/primes"
    workspace = result.workspace
    assert (workspace / "results/primes.txt").read_text(encoding="utf-8").startswith("25 primes")
    assert "import sys" in (workspace / "primes.py").read_text(encoding="utf-8")
    isolation = json.loads((result.archive / "isolation.json").read_text(encoding="utf-8"))
    assert isolation["tier"] == "container"
    assert verify_archive(result.archive) == []
    assert backend.running(result.run_id) == []


PROBE = r'''
import json, os, socket, pathlib
report = {}
try:
    socket.create_connection(("1.1.1.1", 53), timeout=3)
    report["network"] = "reachable"
except OSError as error:
    report["network"] = "blocked"
for target in ("/world/escape.txt", "/escape.txt", "/usr/escape.txt"):
    try:
        pathlib.Path(target).write_text("x")
        report[target] = "written"
    except OSError:
        report[target] = "refused"
report["host_drive"] = os.path.exists("/mnt/c")
report["uid"] = os.getuid()
report["env"] = sorted(os.environ)
report["cwd"] = os.getcwd()
pathlib.Path("inside.txt").write_text("workspace is writable")
print(json.dumps(report))
'''


def test_agent_code_is_confined_by_the_container(habitat_root, tmp_path, backend):  # noqa: F811
    result = run_experiment(habitat_config(tmp_path, [
        {"tool_calls": [call("python.run", code=PROBE)]},
        {"tool_calls": [call("python.run", code="import time\nwhile True: time.sleep(1)",
                             timeout_s=2)]},
    ]), habitat_root)
    assert result.state == "completed", result.reason
    probe, spin = [r["result"] for r in results(result.archive)]
    assert probe["exit_code"] == 0, probe["stderr"]
    report = json.loads(probe["stdout"])
    assert report["network"] == "blocked"
    assert {report[k] for k in ("/world/escape.txt", "/escape.txt", "/usr/escape.txt")} == {
        "refused"}
    assert report["host_drive"] is False
    assert report["cwd"] == "/workspace"
    assert not [name for name in report["env"] if name.startswith("OPEN_LAB")]
    assert (result.workspace / "inside.txt").read_text() == "workspace is writable"
    assert spin["timed_out"] is True and spin["exit_code"] is None
    assert verify_archive(result.archive) == []
    assert backend.running(result.run_id) == []


def test_python_run_is_refused_without_the_subprocess_permission(
        habitat_root, tmp_path, backend):  # noqa: F811
    result = run_experiment(habitat_config(tmp_path, [
        {"tool_calls": [call("python.run", code="print(1)")]},
        {"tool_calls": [call("workspace.edit", path="j.md", append_text="still usable\n")]},
    ], subprocess=False), habitat_root)
    refused, edited = results(result.archive)
    assert refused["status"] != "ok" and "subprocess" in json.dumps(refused)
    assert edited["status"] == "ok"
    assert verify_archive(result.archive) == []


def test_subprocess_cannot_be_granted_on_a_process_tier_world(root, make_config):
    with pytest.raises(LaunchError, match="container isolation tier"):
        run_experiment(make_config([], permissions={"filesystem": "workspace_write",
                                                    "network": "none", "subprocess": True}),
                       root)
