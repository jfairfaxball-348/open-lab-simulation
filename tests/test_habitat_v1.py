"""Integrated Habitat v1 acceptance on the real container backend."""
import json
import shutil
from pathlib import Path

from conftest import events_of
from open_lab_simulation.archive import verify_archive
from open_lab_simulation.supervisor import resume_experiment, run_experiment
from test_container import backend  # noqa: F401

REPO = Path(__file__).resolve().parent.parent
PARENT = REPO / "experiments/habitat/s019-parent-demo.json"
CHILD = REPO / "experiments/habitat/s019-child-demo.json"
RELEASE = REPO / "tests/fixtures/worlds/habitat/1.0.0"


def results(archive):
    found = [e["payload"] for e in events_of(archive) if e["type"] == "tool.result"]
    assert all(r["status"] == "ok" for r in found), found
    return [r["result"] for r in found]


def check_pair(parent, child):
    """Check outcomes and authority boundaries, including intentional program failure."""
    assert parent.state == child.state == "completed"
    assert verify_archive(parent.archive) == verify_archive(child.archive) == []
    values = results(parent.archive)
    assert len(values) == 33
    assert "python.run" in [m["id"] for m in values[0]["matches"]]
    assert values[3]["tick"] == 0
    assert values[7]["exit_code"] == 1 and "NameError" in values[7]["stderr"]
    assert "vlaues" in values[8]["text"]
    assert values[10]["exit_code"] == 0 and values[10]["stdout"] == "10\n"
    assert values[12]["skills"][0]["id"] == "sum-check"
    assert values[13]["skill"] == values[26]["skill"]
    assert values[15]["provenance"] == {"origin": "agent", "created_in_run": parent.run_id}
    assert "agent.double" in [m["id"] for m in values[16]["matches"]]
    assert json.loads(values[17]["documentation"])["provenance"] == values[15]["provenance"]
    assert values[18] == {"value": 14} and values[27] == {"value": 20}
    assert values[21] == values[25] and values[21]["tick"] == 4
    assert values[21]["cells"] != values[3]["cells"]
    assert values[22]["remaining_ms"] > values[24]["remaining_ms"] > 0
    assert values[28]["exit_code"] == 0 and values[29]["text"] == "10"
    assert values[30]["entries"][0]["key"] == "project/plan"
    assert values[31]["exit_code"] == 0 and values[31]["stdout"] == "['journal.md']\n"
    assert values[32]["text"] == "journal.md"
    events = events_of(parent.archive)
    stages = [(i, e["payload"].get("extensions", {}).get("stage"))
              for i, e in enumerate(events) if e["type"] == "supervisor.action"]
    scheduled = next(i for i, stage in stages if stage == "time_sleep_scheduled")
    woke = next(i for i, stage in stages if stage == "time_wake")
    assert woke > scheduled
    assert not any(e["type"] in {"model.call", "agent.output", "environment.message"}
                   for e in events[scheduled + 1:woke])
    inherited = results(child.archive)
    assert len(inherited) == 11
    assert "Scripted plan" in inherited[0]["text"]
    assert "sum(values)" in inherited[1]["text"]
    assert inherited[2]["entries"] == values[30]["entries"]
    assert inherited[3]["skill"] == values[13]["skill"]
    assert inherited[4]["text"] == "10"
    # File/memory continuity does not import the parent's private field or tool authority.
    assert inherited[5]["tick"] == 0 and all(c["drive"] == 0
                                             for c in inherited[5]["cells"])
    assert inherited[6]["provenance"] == {"origin": "agent", "created_in_run": child.run_id}
    assert inherited[7] == {"value": 120}
    assert inherited[8]["exit_code"] == 0 and inherited[9]["text"] == "60"
    assert (parent.workspace / "result.txt").read_text() == "10"
    assert (child.workspace / "child-result.txt").read_text() == "60"
    assert (child.workspace / "notes/index.txt").read_text() == "journal.md"
    assert (child.archive / "lineage.json").is_file()


def test_v1_freedoms_work_together_across_resume_and_data_lineage(tmp_path, backend):  # noqa: F811
    root = tmp_path / "data"
    shutil.copytree(RELEASE, root / "worlds/habitat/1.0.0")
    config = json.loads(PARENT.read_text(encoding="utf-8"))
    assert config["orchestrator"]["provider_id"] == "scripted"
    assert config["permissions"]["network"] == "none"
    paused = run_experiment(PARENT, root, pause_after_steps=24)
    assert paused.state == "paused", paused.reason
    assert verify_archive(paused.archive) == []
    prefix = (paused.archive / "events.jsonl").read_bytes()
    parent = resume_experiment(paused.archive, root)
    assert (parent.archive / "events.jsonl").read_bytes().startswith(prefix)
    original = {name: (parent.archive / name).read_bytes()
                for name in ("events.jsonl", "summary.json")}
    child = run_experiment(CHILD, root, from_run=parent.archive)
    check_pair(parent, child)
    assert all((parent.archive / name).read_bytes() == data for name, data in original.items())
    assert len(list((parent.archive / "children").glob("*.json"))) == 1
    assert backend.running(parent.run_id) == backend.running(child.run_id) == []
