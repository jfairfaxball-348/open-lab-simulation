"""The record's integrity, the world check, and the repository's own boundaries."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest

from conftest import FIXTURE_WORLD, REPO, call, events_of
from open_lab_simulation.archive import verify_archive
from open_lab_simulation.artifacts import WorkspaceScanner, read_index
from open_lab_simulation.contracts import (
    Contracts,
    canonical_json,
    jsonl_lines,
    validate_event_stream,
)
from open_lab_simulation.paths import data_root, minimal_environment
from open_lab_simulation.recorder import Recorder, RecordError
from open_lab_simulation.supervisor import run_experiment
from open_lab_simulation.toolhost import permission_shortfall
from open_lab_simulation.world import WorldError, compute_content_digest, load_world

CONTRACTS = Contracts(FIXTURE_WORLD / "contracts")
AGENT = {"kind": "agent", "model_id": "scripted"}


@pytest.fixture
def recorder(tmp_path) -> Recorder:
    made = Recorder(tmp_path / "events.jsonl", CONTRACTS, "exp-1", "run-1")
    yield made
    made.close()


def output(recorder: Recorder, text: str) -> dict:
    return recorder.record(recorder="agent_runtime", type="agent.output", actor=AGENT,
                           payload={"text": text})


# --- world release ---------------------------------------------------------------

def test_our_digest_agrees_with_the_builder(tmp_path):
    """The fixture was built by the Builder; this is an independent implementation."""
    world = load_world(FIXTURE_WORLD)
    assert compute_content_digest(FIXTURE_WORLD, world.manifest) == world.digest
    assert sorted(world.capabilities) == [
        "math.evaluate", "memory.list", "memory.search", "memory.write",
        "text.stats", "time.now", "workspace.list", "workspace.write"
    ]


@pytest.mark.parametrize(
    "relative",
    ["capabilities/math.evaluate/math_evaluate.py", "contracts/v1/common.schema.json",
     "models/ollama-local.json", "capabilities/text.stats/capability.json"],
)
def test_one_changed_byte_anywhere_is_detected(tmp_path, relative):
    release = Path(shutil.copytree(FIXTURE_WORLD, tmp_path / "release"))
    data = bytearray((release / relative).read_bytes())
    data[data.index(b" ")] = ord("\t")
    (release / relative).write_bytes(bytes(data))
    with pytest.raises(WorldError):
        load_world(release)


def test_added_or_missing_files_are_detected(tmp_path):
    release = Path(shutil.copytree(FIXTURE_WORLD, tmp_path / "release"))
    (release / "capabilities" / "text.stats" / "extra.py").write_text("x = 1\n")
    with pytest.raises(WorldError, match="content digest"):
        load_world(release)
    (release / "capabilities" / "text.stats" / "extra.py").unlink()
    load_world(release)
    (release / "models" / "ollama-local.json").unlink()
    with pytest.raises(WorldError):
        load_world(release)


def test_missing_release_is_an_error_not_a_crash(tmp_path):
    with pytest.raises(WorldError):
        load_world(tmp_path / "nowhere")


# --- recorder --------------------------------------------------------------------

def test_recorder_writes_a_gap_free_hash_chained_log(recorder):
    for text in ("one", "two", "three"):
        output(recorder, text)
    recorder.close()
    lines = jsonl_lines(recorder.path.read_text(encoding="utf-8"))
    assert validate_event_stream(CONTRACTS, lines) == []
    events = [json.loads(line) for line in lines]
    assert [e["seq"] for e in events] == [0, 1, 2]
    assert "prev_event_sha256" not in events[0]
    assert lines[1] == canonical_json(events[1])


def test_recorder_refuses_events_the_contract_forbids(recorder):
    with pytest.raises(RecordError):  # there is no 'agent' recorder
        recorder.record(recorder="agent", type="agent.output", actor=AGENT, payload={"text": "x"})
    with pytest.raises(RecordError):  # a claim cannot be recorded as an observation
        recorder.record(recorder="agent_runtime", type="agent.statement", actor=AGENT,
                        payload={"kind": "goal", "text": "to persist"})
    with pytest.raises(RecordError):  # nor an observation as a claim
        recorder.record(recorder="agent_runtime", type="agent.output", actor=AGENT,
                        epistemic_status="agent_claim", payload={"text": "x"})
    with pytest.raises(RecordError):  # enforcement is the supervisor's to record
        recorder.record(recorder="agent_runtime", type="supervisor.action",
                        payload={"action": "terminated"})
    assert recorder.count == 0 and recorder.path.read_text(encoding="utf-8") == ""
    output(recorder, "still usable")
    assert recorder.count == 1


def test_recorder_never_appends_to_an_existing_log(recorder):
    with pytest.raises(FileExistsError):
        Recorder(recorder.path, CONTRACTS, "exp-1", "run-2")


@pytest.mark.parametrize(
    "tamper",
    [
        lambda lines: lines[:1] + lines[2:],                      # remove an event
        lambda lines: [lines[1], lines[0], *lines[2:]],           # reorder
        lambda lines: [*lines[:1], lines[1].replace("two", "2"), *lines[2:]],   # edit
        lambda lines: lines[:-1] + [lines[-1].replace('"seq":3', '"seq":4')],
        lambda lines: [*lines, lines[-1]],                        # replay
        lambda lines: [json.dumps(json.loads(line)) for line in lines],        # re-serialise
    ],
)
def test_any_edit_to_the_log_is_detected(recorder, tamper):
    for text in ("one", "two", "three", "four"):
        output(recorder, text)
    recorder.close()
    lines = jsonl_lines(recorder.path.read_text(encoding="utf-8"))
    assert validate_event_stream(CONTRACTS, tamper(lines)) != []


# --- archive ---------------------------------------------------------------------

@pytest.fixture
def archive(root, make_config) -> Path:
    config = make_config([
        {"text": "write", "tool_calls": [call("file.write", path="a.txt", content="kept")]},
    ])
    return run_experiment(config, root).archive


def test_archive_tampering_is_detected(archive):
    assert verify_archive(archive) == []
    (record,) = read_index(archive)
    stored = archive / record["archive_path"]

    stored.write_text("changed")
    assert any("does not match its digest" in p for p in verify_archive(archive))
    stored.unlink()
    assert any("content is missing" in p for p in verify_archive(archive))


def test_truncated_or_edited_log_fails_archive_verification(archive):
    log = archive / "events.jsonl"
    lines = jsonl_lines(log.read_text(encoding="utf-8"))
    log.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8", newline="\n")
    assert verify_archive(archive)
    log.write_text("\n".join(lines).replace("write", "wrote") + "\n", encoding="utf-8",
                   newline="\n")
    assert any("hash chain broken" in p for p in verify_archive(archive))
    log.unlink()
    assert verify_archive(archive)


# --- workspace scanner -----------------------------------------------------------

def test_scanner_ignores_links_and_skips_oversized_content(tmp_path, recorder):
    workspace, archive = tmp_path / "ws", tmp_path / "archive"
    workspace.mkdir()
    archive.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("host file")
    (workspace / "big.bin").write_bytes(b"x" * 100)
    (workspace / "small.txt").write_text("ok")
    try:
        (workspace / "link.txt").symlink_to(outside)
    except OSError:
        pass  # creating links needs a privilege on Windows; the rest still applies
    scanner = WorkspaceScanner(workspace, archive, recorder, CONTRACTS, AGENT,
                               max_preserved_bytes=10)
    assert scanner.scan({"kind": "system"}) == 2
    assert scanner.scan({"kind": "system"}) == 0, "an unchanged workspace records nothing"
    records = {r["workspace_path"]: r for r in read_index(archive)}
    assert sorted(records) == ["big.bin", "small.txt"]
    assert records["big.bin"]["preserved"] is False and "archive_path" not in records["big.bin"]
    assert (archive / records["small.txt"]["archive_path"]).read_text() == "ok"
    assert "host file" not in "".join(
        p.read_text(errors="ignore") for p in archive.rglob("*") if p.is_file()
    )


# --- small rules -----------------------------------------------------------------

@pytest.mark.parametrize(
    "needed, granted, shortfall",
    [
        ({"filesystem": "none", "network": "none"},
         {"filesystem": "none", "network": "none"}, None),
        ({"filesystem": "workspace_read", "network": "none"},
         {"filesystem": "workspace_write", "network": "none"}, None),
        ({"filesystem": "workspace_write", "network": "none"},
         {"filesystem": "workspace_read", "network": "none"}, "filesystem 'workspace_write'"),
        ({"filesystem": "none", "network": "loopback"},
         {"filesystem": "none", "network": "none"}, "network 'loopback'"),
        ({"filesystem": "none", "network": "none", "gpu": True},
         {"filesystem": "none", "network": "none"}, "gpu"),
        ({"filesystem": "none", "network": "none", "subprocess": True},
         {"filesystem": "none", "network": "none", "gpu": True}, "subprocess"),
        ({"filesystem": "none", "network": "none", "models": ["vision"]},
         {"filesystem": "none", "network": "none", "models": ["speech"]}, "model 'vision'"),
    ],
)
def test_permission_comparison(needed, granted, shortfall):
    assert permission_shortfall(needed, granted) == shortfall


def test_data_root_and_child_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("OPEN_LAB_DATA", str(tmp_path))
    monkeypatch.setenv("SOME_API_KEY", "secret")
    assert data_root() == tmp_path
    monkeypatch.delenv("OPEN_LAB_DATA")
    assert data_root() == Path.home() / "open-lab-data"
    environment = minimal_environment({"OPEN_LAB_WORKSPACE": "w"})
    assert "SOME_API_KEY" not in environment and "PATH" not in environment
    assert environment["OPEN_LAB_WORKSPACE"] == "w"


def test_simulation_contains_no_builder_code_or_schemas():
    sources = list((REPO / "src").rglob("*.py")) + list((REPO / "tests").glob("*.py"))
    assert sources
    pattern = re.compile(r"^\s*(?:from|import)\s+open_lab_builder", re.MULTILINE)
    assert [p.name for p in sources if pattern.search(p.read_text(encoding="utf-8"))] == []
    assert not list((REPO / "src").rglob("*.schema.json")), "schemas come from the world release"


def test_scripted_events_are_labelled_as_scripted(root):
    result = run_experiment(REPO / "experiments" / "scripted-smoke.json", root)
    for event in events_of(result.archive):
        if event.get("actor", {}).get("kind") == "agent":
            assert event["actor"]["model_id"] == "scripted"
