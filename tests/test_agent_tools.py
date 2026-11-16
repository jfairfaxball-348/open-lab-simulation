"""Pinned agent tools: validation, permissions and real container acceptance."""
import copy
import json
import os
import shutil
from pathlib import Path

import pytest

from conftest import call, events_of
from open_lab_simulation.agent_tools import AgentTools, safe_schema
from open_lab_simulation.archive import verify_archive
from open_lab_simulation.registry import Registry
from open_lab_simulation.supervisor import LaunchError, resume_experiment, run_experiment
from open_lab_simulation.toolhost import ToolHost
from open_lab_simulation.world import load_world
from test_container import backend  # noqa: F401

REPO = Path(__file__).resolve().parent.parent
RELEASE = REPO / "tests/fixtures/worlds/habitat/0.3.0"
DEMO = REPO / "experiments/habitat/s016b-demo.json"
CONFIG = json.loads(DEMO.read_text(encoding="utf-8"))
DEFINITION = CONFIG["extensions"]["scripted_model"]["replies"][3]["tool_calls"][0][
    "arguments"]["arguments"]
SOURCE = 'def run(arguments):\n    return {"value": arguments["value"] * 2}\n'


@pytest.fixture
def tool_root(tmp_path):
    root = tmp_path / "data"
    shutil.copytree(RELEASE, root / "worlds/habitat/0.3.0")
    return root


def outcomes(archive):
    return [e["payload"] for e in events_of(archive) if e["type"] == "tool.result"]


def test_demo_validates_pins_rediscovers_and_resumes(tool_root, backend):  # noqa: F811
    paused = run_experiment(DEMO, tool_root, pause_after_steps=4)
    assert paused.state == "paused", paused.reason
    before = (paused.archive / "events.jsonl").read_bytes()
    assert verify_archive(paused.archive) == []
    resumed = resume_experiment(paused.archive, tool_root)
    assert resumed.state == "completed", resumed.reason
    found = outcomes(resumed.archive)
    assert len(found) == 9 and all(r["status"] == "ok" for r in found), found
    values = [r["result"] for r in found]
    assert "tools.create" in [m["id"] for m in values[0]["matches"]]
    created = values[3]
    assert created["registered"] is True and created["examples_passed"] == 1
    assert created["provenance"] == {"origin": "agent", "created_in_run": resumed.run_id}
    assert values[5]["matches"][0]["id"] == "agent.double"
    assert values[5]["matches"][0]["provenance"] == created["provenance"]
    described = json.loads(values[6]["documentation"])
    assert described["provenance"] == created["provenance"]
    assert described["extensions"]["source_sha256"] == created["source_sha256"]
    assert values[7] == {"value": 14}
    assert found[7]["extensions"]["source_sha256"] == created["source_sha256"]
    assert values[8]["matches_expected"] is True
    assert "changed workspace source" in (resumed.workspace / "double.py").read_text()
    assert (resumed.archive / "events.jsonl").read_bytes().startswith(before)
    assert verify_archive(resumed.archive) == []
    assert backend.running(resumed.run_id) == []


@pytest.mark.parametrize("code, expected", [
    ('def run(a): return {"value": 999}', "differs"),
    ('def run(a): return {"value": "wrong type"}', "schema"),
    ('def run(a): raise RuntimeError("failure")', "execution failed"),
    ('def run(a):\n    while True: pass', "timed out"),
    ('def run(a):\n    print("extra")\n    return {"value": 6}', "JSON"),
    ('def run(a):\n    print("x" * 300000)\n    return {"value": 6}', "truncated"),
])
def test_failed_examples_never_register(tool_root, tmp_path, backend, code, expected):  # noqa: F811
    config = copy.deepcopy(CONFIG)
    config.pop("checkpointing")
    config["extensions"]["scripted_model"]["replies"] = [
        {"tool_calls": [call("workspace.write", path="double.py", text=code)]},
        {"tool_calls": [call("tools.create", **DEFINITION)]},
        {"tool_calls": [{"capability_id": "capabilities.describe",
                         "arguments": {"id": "agent.double"}}]},
    ]
    path = tmp_path / "failure.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    result = run_experiment(path, tool_root)
    assert result.state == "completed", result.reason
    found = outcomes(result.archive)
    assert found[1]["status"] == "error" and expected in found[1]["error"], found
    assert found[2]["status"] == "error" and "unknown capability" in found[2]["error"]
    assert verify_archive(result.archive) == []
    assert backend.running(result.run_id) == []


def test_agent_tool_code_cannot_escape_during_validation(tool_root, tmp_path, backend):  # noqa: F811
    config = copy.deepcopy(CONFIG)
    config.pop("checkpointing")
    definition = {**DEFINITION, "examples": [{"input": {"value": 3},
                                             "expected_output": {"value": 6}}]}
    code = '''import os, socket
from pathlib import Path
def run(a):
    assert not Path('/mnt/c').exists()
    for name in ('/world/escape', '/escape', '/usr/escape'):
        try:
            Path(name).write_text('bad')
        except OSError:
            pass
        else:
            raise AssertionError('writable boundary')
    try:
        socket.create_connection(('1.1.1.1', 53), timeout=1)
    except OSError:
        pass
    else:
        raise AssertionError('network reachable')
    assert not any(k.startswith('OPEN_LAB') for k in os.environ)
    Path('validation-side-effect.txt').write_text('sandbox only')
    return {'value': a['value'] * 2}
'''
    config["extensions"]["scripted_model"]["replies"] = [
        {"tool_calls": [call("workspace.write", path="double.py", text=code)]},
        {"tool_calls": [call("tools.create", **definition)]},
        {"tool_calls": [call("agent.double", value=9)]}]
    path = tmp_path / "isolation.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    result = run_experiment(path, tool_root)
    assert result.state == "completed", result.reason
    found = outcomes(result.archive)
    assert all(r["status"] == "ok" for r in found), found
    assert found[-1]["result"] == {"value": 18}
    assert (result.workspace / "validation-side-effect.txt").read_text() == "sandbox only"
    assert verify_archive(result.archive) == []


@pytest.fixture
def unit_host(tmp_path):
    world = load_world(RELEASE)
    # Unit tests never execute source. Live tests above exercise the actual runner.
    host = ToolHost(dict(world.capabilities), tmp_path, CONFIG["permissions"], lambda event: None,
                    {"kind": "agent", "model_id": "scripted"},
                    registry=Registry(RELEASE, world.capabilities))
    host.sandbox = object()
    host.agent_tools = AgentTools(host, "run-unit", world.contracts)
    (tmp_path / "double.py").write_text(SOURCE, encoding="utf-8")
    return host


def test_duplicate_and_builder_ids_cannot_register(unit_host):
    tools = unit_host.agent_tools
    tools.install(DEFINITION, SOURCE)
    with pytest.raises(ValueError, match="already exists"):
        tools.validate(DEFINITION, SOURCE)
    for ident in ("python.run", "capabilities.invoke", "tools.create"):
        with pytest.raises(ValueError, match="definition"):
            tools.validate({**DEFINITION, "id": ident}, SOURCE)


def test_permissions_and_opt_in_apply_before_source_or_execution(unit_host):
    unit_host.agent_tools.install(DEFINITION, SOURCE)
    for target in ("tools.create", "agent.double"):
        unit_host.deny = {"python.run"}
        assert "python.run unavailable" in unit_host._unavailable(target)
        unit_host.deny = {target}
        assert "not available" in unit_host._unavailable(target)
        unit_host.deny = set()
        unit_host.permissions = {"filesystem": "workspace_read", "network": "none"}
        assert "permission" in unit_host._unavailable(target)
        unit_host.permissions = CONFIG["permissions"]
    unit_host.agent_tools = None
    assert "not enabled" in unit_host._unavailable("tools.create")


@pytest.mark.parametrize("schema", [
    {"type": "object", "$ref": "https://example.org/schema"},
    {"type": "object", "additionalProperties": False, "patternProperties": {".*": {}}},
    {"type": "object", "additionalProperties": True},
    {"type": "array", "items": {"type": "integer"}, "maxItems": 1025},
    {"type": "string", "maxLength": 16001},
    {"type": "object", "additionalProperties": False, "required": "incorrect"},
])
def test_unsafe_or_invalid_schema_is_refused(schema):
    with pytest.raises(ValueError):
        safe_schema(schema)


def test_source_paths_links_and_size_are_refused(unit_host, tmp_path):
    tools = unit_host.agent_tools
    for value in ("../double.py", "/double.py", "C:/double.py", "double.txt", "a\\double.py",
                  "con.py", "COM1.py"):
        with pytest.raises(ValueError):
            tools.source(value)
    (tmp_path / "big.py").write_bytes(b"x" * 32001)
    with pytest.raises(ValueError, match="32000"):
        tools.source("big.py")
    os.link(tmp_path / "double.py", tmp_path / "linked.py")
    with pytest.raises(ValueError, match="hard links"):
        tools.source("linked.py")
    directory = tmp_path / "outside"
    directory.mkdir()
    if os.name == "nt":
        import _winapi

        _winapi.CreateJunction(str(directory), str(tmp_path / "link"))
    else:
        (tmp_path / "link").symlink_to(directory, target_is_directory=True)
    with pytest.raises(ValueError, match="links"):
        tools.source("link/double.py")


def test_restore_rejects_changed_snapshot_without_executing_examples(unit_host):
    tools = unit_host.agent_tools
    tools.install(DEFINITION, SOURCE)
    record = copy.deepcopy(tools.records)
    record["agent.double"]["source"] = "raise RuntimeError('changed')"
    world = load_world(RELEASE)
    clean = ToolHost(dict(world.capabilities), unit_host.workspace, CONFIG["permissions"],
                     lambda event: None, {"kind": "system"})
    with pytest.raises(ValueError, match="digest"):
        AgentTools(clean, "run-unit", world.contracts, record)


def test_capacity_and_schema_limits_refuse_before_execution(unit_host):
    tools = unit_host.agent_tools
    for n in range(16):
        tools.install({**DEFINITION, "id": f"agent.double-{n}"}, SOURCE)
    with pytest.raises(ValueError, match="sixteen"):
        tools.create({**DEFINITION, "id": "agent.seventeenth"})
    schema = {"type": "integer"}
    for _ in range(10):
        schema = {"type": "object", "properties": {"nested": schema},
                  "additionalProperties": False}
    with pytest.raises(ValueError, match="depth"):
        safe_schema(schema)
    with pytest.raises(ValueError, match="16000"):
        safe_schema({"type": "object", "description": "x" * 16001})


def test_bad_example_definition_is_not_executed_or_installed(unit_host):
    arguments = {**DEFINITION, "examples": [{"input": {"value": "wrong"},
                                            "expected_output": {"value": 6}}]}
    with pytest.raises(ValueError, match="schema"):
        unit_host.agent_tools.create(arguments)
    assert not unit_host.agent_tools.records


def test_enablement_requires_container_and_subprocess(tool_root, tmp_path):
    config = copy.deepcopy(CONFIG)
    config["permissions"]["subprocess"] = False
    path = tmp_path / "no-permission.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(LaunchError, match="container isolation and subprocess"):
        run_experiment(path, tool_root)
