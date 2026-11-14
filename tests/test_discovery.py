"""Discovery through only four tools, with real execution and independent records."""
from __future__ import annotations

import json

import pytest

from conftest import FIXTURE_WORLD, events_of, restamp, tree
from open_lab_simulation.archive import verify_archive
from open_lab_simulation.artifacts import read_index
from open_lab_simulation.model import fit_context, parse_reply, tool_definitions
from open_lab_simulation.registry import META_TOOLS, PREAMBLE, Registry
from open_lab_simulation.supervisor import LaunchError, run_experiment
from open_lab_simulation.world import load_world


def meta(name, **arguments):
    return {"capability_id": f"capabilities.{name}", "arguments": arguments}


def outcomes(archive):
    return [e["payload"] for e in events_of(archive) if e["type"] == "tool.result"]


def test_scripted_discovery_sequence_and_demo_really_write_files(root, make_config):
    replies = [{"tool_calls": [tool]} for tool in (
        meta("search", query="save notes", category="workspace"),
        meta("describe", id="workspace.write"),
        meta("demonstrate", id="workspace.write"),
        meta("invoke", id="workspace.write", arguments={"path": "found.txt", "text": "found"}),
    )]
    result = run_experiment(make_config(replies), root)
    assert result.state == "completed"
    found, described, demo, used = outcomes(result.archive)
    assert found["result"]["matches"][0]["id"] == "workspace.write"
    doc = described["result"]["documentation"]
    assert described["result"]["example_call"] == {
        "tool": "capabilities.invoke", "arguments": {
            "id": "workspace.write", "arguments": {
                "path": "notes/hello.txt", "text": "Hello laboratory!\n"}}}
    assert described["result"]["input_schema"]["required"] == ["path", "text"]
    for field in ("input_schema", "output_schema", "permissions", "limitations", "failure_modes"):
        assert f"## {field}" in doc
    assert demo["result"]["input"] == {"path": "notes/hello.txt", "text": "Hello laboratory!\n"}
    assert demo["result"]["outcome"] == {"status": "ok", "result": {
        "path": "notes/hello.txt", "bytes_written": 18}}
    assert demo["result"]["matches_expected"] is True
    assert used["result"] == {"path": "found.txt", "bytes_written": 5}
    assert (result.workspace / "notes/hello.txt").read_bytes() == b"Hello laboratory!\n"
    assert (result.workspace / "found.txt").read_bytes() == b"found"
    artifacts = read_index(result.archive)
    assert len(artifacts) == 2
    assert all(a["producer"]["capability_id"] == "workspace.write" for a in artifacts)
    calls = [e for e in events_of(result.archive) if e["type"] == "tool.call"]
    assert [e["payload"]["capability_id"] for e in calls] == [
        "capabilities.search", "capabilities.describe", "capabilities.demonstrate",
        "capabilities.invoke"]
    assert verify_archive(result.archive) == []


def test_named_demo_real_clock_and_failed_demo_are_reported(root, make_config):
    replies = [{"tool_calls": [
        meta("demonstrate", id="text.stats", demo_id="empty"),
        meta("demonstrate", id="time.now"),
        meta("demonstrate", id="workspace.write"),
        meta("demonstrate", id="workspace.write"),  # second exclusive creation fails
    ]}]
    result = run_experiment(make_config(replies), root)
    answers = outcomes(result.archive)
    assert answers[0]["result"]["demo_id"] == "empty"
    assert answers[0]["result"]["outcome"]["result"]["words"] == 0
    assert answers[1]["result"]["outcome"]["result"]["utc"].endswith("Z")
    assert "expected_output" not in answers[1]["result"]
    assert answers[3]["result"]["outcome"]["status"] == "error"
    assert answers[3]["result"]["matches_expected"] is False
    assert verify_archive(result.archive) == []


@pytest.mark.parametrize("changes", [
    {"capabilities": {"deny": ["workspace.write"]}},
    {"capabilities": {"allow": ["text.stats"]}},
    {"permissions": {"filesystem": "workspace_read"}},
])
def test_search_describe_demo_and_invoke_cannot_widen_authority(root, make_config, changes):
    result = run_experiment(make_config([{"tool_calls": [
        meta("search", query="write"), meta("describe", id="workspace.write"),
        meta("demonstrate", id="workspace.write"),
        meta("invoke", id="workspace.write", arguments={"path": "no", "text": "no"}),
    ]}], **changes), root)
    search, *denied = outcomes(result.archive)
    assert "workspace.write" not in {m["id"] for m in search["result"]["matches"]}
    assert [o["status"] for o in denied] == ["denied"] * 3
    assert list(result.workspace.iterdir()) == []
    assert verify_archive(result.archive) == []


def test_invalid_meta_calls_unknown_ids_and_direct_calls_are_recorded(root, make_config):
    tools = [meta("search", query=4), meta("describe", id="../outside"),
             meta("invoke", id="text.stats", arguments={"text": 9}),
             meta("invoke", id="capabilities.invoke", arguments={}),
             meta("demonstrate", id="text.stats", demo_id="absent"),
             meta("describe", id="absent"), meta("search", query="", extra=True),
             {"capability_id": "text.stats", "arguments": {"text": "unadvertised"}}]
    result = run_experiment(make_config([{"tool_calls": tools}]), root)
    assert [o["status"] for o in outcomes(result.archive)] == ["error"] * len(tools)
    assert list(result.workspace.iterdir()) == []
    assert verify_archive(result.archive) == []


def test_search_protocol_relevance_case_filter_empty_and_ties():
    world = load_world(FIXTURE_WORLD)
    registry = Registry(world.directory, world.capabilities)
    available = set(world.capabilities)
    for query, ident in (("count words", "text.stats"), ("fractions arithmetic", "math.evaluate"),
                         ("save notes", "workspace.write"), ("browse folder", "workspace.list"),
                         ("CURRENT utc CLOCK", "time.now")):
        assert registry.search(query, available)[0]["id"] == ident
    assert registry.search("???", available) == registry.search("missingword", available) == []
    assert registry.search("write", available, "time") == []
    entry = registry.entries["workspace.write"]
    registry.entries = {"b.tool": entry, "a.tool": entry}
    assert [m["id"] for m in registry.search("write", {"b.tool", "a.tool"})] == ["a.tool", "b.tool"]


@pytest.mark.parametrize("mutation", ["missing", "version", "tokens", "doc", "duplicate"])
def test_bad_registry_is_refused_before_any_run_is_created(root, release, make_config, mutation):
    path = release / "registry/index.json"
    index = json.loads(path.read_text(encoding="utf-8"))
    if mutation == "missing":
        path.unlink()
    else:
        if mutation == "version":
            index["format_version"] = "99.0.0"
        elif mutation == "tokens":
            index["capabilities"][0]["tokens"]["name"] = []
        elif mutation == "doc":
            index["capabilities"][0]["documentation_path"] = "../escape.txt"
        else:
            index["capabilities"].append(index["capabilities"][0])
        path.write_text(json.dumps(index), encoding="utf-8")
    restamp(release)
    before = tree(root)
    with pytest.raises(LaunchError, match="registry"):
        run_experiment(make_config([]), root)
    assert tree(root) == before and not (root / "runs").exists()


def test_preamble_and_tools_contain_no_specific_capability_and_are_recorded(root, make_config):
    definitions, _ = tool_definitions(META_TOOLS)
    world = load_world(FIXTURE_WORLD)
    for ident in world.capabilities:
        assert ident not in PREAMBLE and ident not in json.dumps(definitions)
    result = run_experiment(make_config([]), root)
    preambles = [e for e in events_of(result.archive) if e["type"] == "environment.message"]
    assert len(preambles) == 1
    assert preambles[0]["payload"] == {"kind": "preamble", "text": PREAMBLE}
    assert preambles[0]["epistemic_status"] == "observed"
    assert verify_archive(result.archive) == []


def test_context_guard_preserves_preamble_and_initial_prompt():
    messages = [{"role": "system", "content": PREAMBLE}, {"role": "user", "content": "premise"},
                {"role": "assistant", "content": "old" * 1000},
                {"role": "tool", "content": "old result"}, {"role": "user", "content": "latest"}]
    assert fit_context(messages, 0, 1)[1] == 2
    assert messages == [{"role": "system", "content": PREAMBLE},
                        {"role": "user", "content": "premise"},
                        {"role": "user", "content": "latest"}]


def test_world_without_preamble_contract_support_is_refused(root, release, make_config):
    path = release / "contracts/v1/observation-event.schema.json"
    schema = json.loads(path.read_text(encoding="utf-8"))
    schema["x-schema-version"] = "1.1.0"
    for rule in schema["allOf"]:
        kinds = rule.get("then", {}).get("properties", {}).get("payload", {}).get(
            "properties", {}).get("kind", {}).get("enum", [])
        if "preamble" in kinds:
            kinds.remove("preamble")
    path.write_text(json.dumps(schema), encoding="utf-8")
    restamp(release, lambda m: m["contracts"].update({"observation-event": "1.1.0"}))
    before = tree(root)
    with pytest.raises(LaunchError, match="preamble"):
        run_experiment(make_config([]), root)
    assert tree(root) == before and not (root / "runs").exists()


def test_new_documentation_reaches_model_before_old_context_is_dropped():
    messages = [{"role": "system", "content": PREAMBLE}, {"role": "user", "content": "premise"},
                {"role": "assistant", "content": "old" * 1000},
                {"role": "tool", "content": "old result"},
                {"role": "assistant", "tool_calls": [{"id": "describe"}], "content": ""},
                {"role": "tool", "content": "full documentation " * 200}]
    _, dropped = fit_context(messages, 0, 5000)
    assert dropped == 2
    assert messages[-1]["content"] == "full documentation " * 200
    assert messages[0]["content"] == PREAMBLE and messages[1]["content"] == "premise"


def test_json_encoded_nested_input_is_repaired_without_rewriting_what_was_said():
    raw = json.dumps({"id": "time.now", "arguments": "{}"})
    message = {"tool_calls": [{"function": {"name": "capabilities_invoke", "arguments": raw}}]}
    _, names = tool_definitions(META_TOOLS)
    _, calls, repaired = parse_reply(message, names)
    assert repaired == 1 and calls[0].arguments == {"id": "time.now", "arguments": {}}
    assert calls[0].raw_arguments == raw
    assert message["tool_calls"][0]["function"]["arguments"] == raw
    message["tool_calls"][0]["function"]["arguments"] = json.dumps(
        {"id": "time.now", "arguments": "not JSON"})
    _, calls, repaired = parse_reply(message, names)
    assert repaired == 0 and calls[0].arguments["arguments"] == "not JSON"


def test_exact_extra_invocation_envelope_is_repaired_but_missing_fields_are_not_invented():
    _, names = tool_definitions(META_TOOLS)
    for inner, expected_repairs in (({"id": "time.now", "arguments": {}}, 1),
                                    ({"arguments": {}}, 0),
                                    ({"id": "time.now", "arguments": {}, "extra": True}, 0)):
        raw = json.dumps({"arguments": inner})
        _, calls, repaired = parse_reply({"tool_calls": [{"function": {
            "name": "capabilities_invoke", "arguments": raw}}]}, names)
        assert repaired == expected_repairs and calls[0].raw_arguments == raw
        assert calls[0].arguments == (inner if expected_repairs else {"arguments": inner})
