"""Use the generated 0.8.0 release through the real supervisor and tool host."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from conftest import call, events_of
from open_lab_simulation.archive import verify_archive
from open_lab_simulation.supervisor import run_experiment
from open_lab_simulation.world import load_world


def world(root):
    source = Path(__file__).parent / "fixtures/worlds/minimal/0.8.0"
    target = root / "worlds/minimal/0.8.0"
    shutil.copytree(source, target)
    loaded = load_world(target)
    assert len(loaded.capabilities) == 11
    return loaded


def test_statistics_discovery_demonstration_and_independent_invocation(root, make_config):
    loaded = world(root)
    tools = [
        {"capability_id": "capabilities.search", "arguments": {"query": "statistics data"}},
        {"capability_id": "capabilities.describe", "arguments": {"id": "stats.describe"}},
        {"capability_id": "capabilities.demonstrate", "arguments": {"id": "stats.describe"}},
        call("stats.describe", values=[2, 4, 6]),
        {"capability_id": "capabilities.describe", "arguments": {"id": "stats.regress"}},
        {"capability_id": "capabilities.demonstrate", "arguments": {"id": "stats.regress"}},
        call("stats.regress", x=[1, 2, 3], y=[3, 5, 7]),
    ]
    config = make_config([{"tool_calls": [tool]} for tool in tools],
                         world={"version": "0.8.0", "content_digest": loaded.digest},
                         permissions={"filesystem": "none", "network": "none"})
    result = run_experiment(config, root)
    assert result.state == "completed", result.notes
    assert verify_archive(result.archive) == []
    answers = [e["payload"] for e in events_of(result.archive) if e["type"] == "tool.result"]
    assert all(a["status"] == "ok" for a in answers)
    assert {"stats.describe", "stats.regress"} <= {
        m["id"] for m in answers[0]["result"]["matches"]}
    assert "divisor n" in answers[1]["result"]["documentation"]
    assert answers[2]["result"]["matches_expected"] is True
    assert answers[3]["result"]["mean"] == 4
    assert answers[5]["result"]["matches_expected"] is True
    assert answers[6]["result"] == {"count": 3, "slope": 2, "intercept": 1, "r": 1}
    assert list(result.workspace.iterdir()) == []


def test_statistics_invalid_and_degenerate_inputs_are_recorded(root, make_config):
    world(root)
    tools = [call("stats.describe", values=[]), call("stats.describe", values=[True]),
             call("stats.describe", values=[1], code="create a file"),
             call("stats.regress", x=[1, 2], y=[1, 2, 3]),
             call("stats.regress", x=[1, 1], y=[1, 2]),
             call("stats.regress", x=[1, 2], y=[7, 7])]
    result = run_experiment(make_config([{"tool_calls": [t]} for t in tools],
                                       world={"version": "0.8.0"},
                                       permissions={"filesystem": "none", "network": "none"}), root)
    assert result.state == "completed", result.notes
    assert verify_archive(result.archive) == []
    answers = [e["payload"] for e in events_of(result.archive) if e["type"] == "tool.result"]
    assert [a["status"] for a in answers] == ["error"] * 5 + ["ok"]
    assert answers[-1]["result"]["r"] is None
    assert list(result.workspace.iterdir()) == []


def test_statistics_deny_list_is_enforced_on_search_and_execution(root, make_config):
    world(root)
    tools = [{"capability_id": "capabilities.search", "arguments": {"query": "statistics"}},
             call("stats.describe", values=[1, 2]), call("stats.regress", x=[1, 2], y=[1, 2])]
    result = run_experiment(make_config([{"tool_calls": [t]} for t in tools],
                                       world={"version": "0.8.0"},
                                       capabilities={"deny": ["stats.describe", "stats.regress"]}),
                            root)
    assert verify_archive(result.archive) == []
    answers = [e["payload"] for e in events_of(result.archive) if e["type"] == "tool.result"]
    assert not {"stats.describe", "stats.regress"} & {
        m["id"] for m in answers[0]["result"]["matches"]}
    assert [a["status"] for a in answers[1:]] == ["denied", "denied"]
    frozen = json.loads((result.archive / "experiment.json").read_text(encoding="utf-8"))
    assert frozen["capabilities"]["deny"] == ["stats.describe", "stats.regress"]
