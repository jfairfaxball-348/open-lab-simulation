"""KI-24: recorded JSONL is split on "\\n" only.

``canonical_json`` writes U+0085, U+2028 and U+2029 raw (``ensure_ascii=False``),
and ``str.splitlines()`` treats each of them as a line break. A reader that used
``splitlines()`` therefore cut one intact event in two and reported a sound log
as broken (S014i-001 cell ``s41-r2-c``).
"""

from __future__ import annotations

import ast
import json

from conftest import REPO, call, events_of
from open_lab_simulation import benchmark, providers
from open_lab_simulation.archive import verify_archive
from open_lab_simulation.artifacts import read_index
from open_lab_simulation.benchmark_analysis import analyse_archive
from open_lab_simulation.contracts import Contracts, canonical_json, jsonl_lines
from open_lab_simulation.recorder import Recorder
from open_lab_simulation.supervisor import run_experiment

SEPARATORS = "line sep para sep next\u0085line"


def test_the_separators_are_written_raw_and_split_by_splitlines():
    line = canonical_json({"text": SEPARATORS})
    assert all(ch in line for ch in "  \u0085")
    assert len(line.splitlines()) == 4
    assert jsonl_lines(line + "\n") == [line]


def test_jsonl_lines_splits_on_newline_only():
    assert jsonl_lines("") == []
    assert jsonl_lines("a\n") == ["a"]
    assert jsonl_lines("a\nb\n") == ["a", "b"]
    assert jsonl_lines("a\n\nb\n") == ["a", "", "b"]
    assert jsonl_lines("a\nb") == ["a", "b"]  # callers decide what a torn line means
    assert jsonl_lines(f"x{SEPARATORS}\n") == [f"x{SEPARATORS}"]


def test_a_run_recording_the_separators_verifies(root, make_config):
    config = make_config([
        {"text": SEPARATORS,
         "tool_calls": [call("text.stats", text=SEPARATORS),
                        call("workspace.write", path="notes.txt", content=SEPARATORS)]},
        {"text": f"done {SEPARATORS}"},
    ])
    result = run_experiment(config, root)
    raw = (result.archive / "events.jsonl").read_text(encoding="utf-8")
    assert all(ch in raw for ch in "  \u0085")
    assert verify_archive(result.archive) == []

    events = events_of(result.archive)
    assert len(events) == raw.count("\n")
    assert [e["seq"] for e in events] == list(range(len(events)))
    outputs = [e["payload"]["text"] for e in events if e["type"] == "agent.output"]
    assert SEPARATORS in outputs
    loaded, _, digest = analyse_archive(result.archive)
    assert loaded == events and len(digest) == 64


def test_a_recorder_appends_to_a_log_holding_the_separators(tmp_path):
    from conftest import FIXTURE_WORLD

    contracts = Contracts(FIXTURE_WORLD / "contracts")
    path = tmp_path / "events.jsonl"
    actor = {"kind": "agent", "model_id": "scripted"}
    first = Recorder(path, contracts, "exp-1", "run-1")
    first.record(recorder="agent_runtime", type="agent.output", actor=actor,
                 payload={"text": SEPARATORS})
    first.close()
    again = Recorder(path, contracts, "exp-1", "run-1", append=True)
    assert again.count == 1
    again.record(recorder="agent_runtime", type="agent.output", actor=actor,
                 payload={"text": "after"})
    again.close()
    assert [e["payload"]["text"] for e in
            map(json.loads, jsonl_lines(path.read_text(encoding="utf-8")))] == [
        SEPARATORS, "after"]


def test_ledgers_and_the_artifact_index_hold_the_separators(tmp_path):
    campaign = tmp_path / "ledger.jsonl"
    benchmark.append_ledger(campaign, kind="note", text=SEPARATORS)
    benchmark.append_ledger(campaign, kind="note", text="second")
    assert [e["text"] for e in benchmark.read_ledger(campaign)] == [SEPARATORS, "second"]

    ledger = providers.Ledger(tmp_path, "fake-hosted")
    mirror = tmp_path / "provider-ledger.jsonl"
    ledger.append({"kind": "reserve", "run_id": "r", "error": SEPARATORS}, mirror)
    ledger.append({"kind": "outcome", "run_id": "r"}, mirror)
    assert [e["kind"] for e in ledger.entries()] == ["reserve", "outcome"]
    assert len(jsonl_lines(mirror.read_text(encoding="utf-8"))) == 2

    archive = tmp_path / "archive"
    (archive / "artifacts").mkdir(parents=True)
    (archive / "artifacts" / "index.jsonl").write_text(
        canonical_json({"path": SEPARATORS}) + "\n" + canonical_json({"path": "b"}) + "\n",
        encoding="utf-8", newline="\n")
    assert [m["path"] for m in read_index(archive)] == [SEPARATORS, "b"]


# Calls of splitlines() that do not read recorded JSONL: subprocess output and a
# GPU query, where every separator is a real line break.
NOT_JSONL = {"container.py", "resources.py", "toolhost.py"}


def test_no_reader_of_recorded_jsonl_uses_splitlines():
    found = []
    for path in sorted((REPO / "src" / "open_lab_simulation").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Attribute) and node.attr == "splitlines"
                    and path.name not in NOT_JSONL):
                found.append(f"{path.name}:{node.lineno}")
    assert found == []
