"""S016a live acceptance: procedural notes remain data across checkpoint resume."""
import json
import shutil
from pathlib import Path

import pytest

from conftest import events_of
from open_lab_simulation.archive import verify_archive
from open_lab_simulation.supervisor import resume_experiment, run_experiment
from test_container import backend  # noqa: F401

REPO = Path(__file__).resolve().parent.parent
DEMO = REPO / "experiments/habitat/s016a-demo.json"
RELEASE = REPO / "tests/fixtures/worlds/habitat/0.2.1"


@pytest.fixture
def skill_root(tmp_path):
    root = tmp_path / "data"
    shutil.copytree(RELEASE, root / "worlds/habitat/0.2.1")
    return root


def outcomes(archive):
    return [e["payload"] for e in events_of(archive) if e["type"] == "tool.result"]


def test_saved_skill_survives_pause_resume_and_is_reused(skill_root, backend):  # noqa: F811
    paused = run_experiment(DEMO, skill_root, pause_after_steps=3)
    assert paused.state == "paused", paused.reason
    assert verify_archive(paused.archive) == []
    saved = (paused.workspace / ".skills/sum-check.json").read_bytes()
    before = (paused.archive / "events.jsonl").read_bytes()
    # Resume restores protected checkpoint bytes, rather than trusting later edits.
    (paused.workspace / ".skills/sum-check.json").write_text("damaged", encoding="utf-8")
    resumed = resume_experiment(paused.archive, skill_root)
    assert resumed.state == "completed", resumed.reason
    assert (resumed.workspace / ".skills/sum-check.json").read_bytes() == saved
    assert (resumed.archive / "events.jsonl").read_bytes().startswith(before)
    found = outcomes(resumed.archive)
    assert len(found) == 7 and all(r["status"] == "ok" for r in found), found
    results = [r["result"] for r in found]
    assert "skills.save" in [m["id"] for m in results[0]["matches"]]
    assert "steps" in results[1]["input_schema"]["properties"]
    assert results[2]["provenance"] == {"origin": "agent"}
    assert results[3]["skills"][0]["id"] == "sum-check"
    assert results[4]["skill"]["examples"] == ["[2, 3, 5] gives 10."]
    assert results[5]["exit_code"] == 0 and results[5]["stdout"] == "10\n"
    assert results[6]["text"] == "10"
    assert verify_archive(resumed.archive) == []
    assert backend.running(resumed.run_id) == []


def test_links_are_refused_in_the_live_container(tmp_path, backend):  # noqa: F811
    from open_lab_simulation.toolhost import Sandbox, ToolHost
    from open_lab_simulation.world import load_world

    world = load_world(RELEASE)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = json.loads(DEMO.read_text(encoding="utf-8"))
    save_args = config["extensions"]["scripted_model"]["replies"][2]["tool_calls"][0][
        "arguments"]["arguments"]
    sandbox = Sandbox(backend, RELEASE, "run-skill-link-test",
                      {"ram_mb": 256, "cpu_cores": 0.5, "max_processes": 16})
    host = ToolHost(world.capabilities, workspace, config["permissions"], lambda event: None,
                    {"kind": "agent", "model_id": "scripted"}, sandbox=sandbox)
    try:
        created = host.call("python.run", {"code": "from pathlib import Path\n"
                            "Path('.skills').symlink_to('/world', target_is_directory=True)"})
        assert created["status"] == "ok" and created["result"]["exit_code"] == 0
        for name, args in (("skills.save", save_args), ("skills.list", {}),
                           ("skills.get", {"id": "sum-check"})):
            refused = host.call(name, args)
            assert refused["status"] != "ok" and "links" in json.dumps(refused)
        created = host.call("python.run", {"code": "from pathlib import Path\n"
                            "Path('.skills').unlink()\nPath('.skills').mkdir()\n"
                            "Path('.skills/sum-check.json').symlink_to('/etc/passwd')"})
        assert created["status"] == "ok" and created["result"]["exit_code"] == 0
        for name, args in (("skills.get", {"id": "sum-check"}),
                           ("skills.save", {**save_args, "replace": True})):
            refused = host.call(name, args)
            assert refused["status"] != "ok" and "links" in json.dumps(refused)
    finally:
        backend.sweep(sandbox.run_id)
    assert backend.running(sandbox.run_id) == []
