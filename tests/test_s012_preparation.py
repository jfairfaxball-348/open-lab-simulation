"""Validate the frozen real-model plan without launching a model or experiment."""
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

from open_lab_simulation.world import load_world

REPO = Path(__file__).resolve().parent.parent
PLAN = REPO / "experiments/s012"
spec = importlib.util.spec_from_file_location("s012_readiness", PLAN / "check_ready.py")
readiness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(readiness)


def test_exact_approved_prompt_and_full_habitat_configuration_are_valid():
    world = load_world(REPO / "tests/fixtures/worlds/habitat/1.0.0")
    config, frozen = readiness.verify_plan(PLAN, world)
    assert config["orchestrator"]["model_id"] == "orchestrator-gpt-5-6-luna"
    assert config["limits"]["wall_clock_s"] == 3600
    assert config["limits"]["max_steps"] == frozen["max_requests"] == 180
    assert config["capabilities"]["agent_created_tools"] is True
    assert config["permissions"]["subprocess"] is True
    assert all(c[1]["permissions"]["network"] == "none" for c in world.capabilities.values())
    assert "recovery" not in config["extensions"]
    assert "reliability_fault" not in config["extensions"]
    assert config["extensions"]["runtime_policy"]["id"] == "self-scheduled-v1"


@pytest.mark.parametrize("name", ["initial-prompt.txt", "first-observation.json"])
def test_changed_prompt_or_configuration_is_refused(tmp_path, name):
    shutil.copytree(PLAN, tmp_path / "plan")
    target = tmp_path / "plan" / name
    target.write_bytes(target.read_bytes() + b"\n")
    world = load_world(REPO / "tests/fixtures/worlds/habitat/1.0.0")
    with pytest.raises(ValueError, match="bytes changed"):
        readiness.verify_plan(tmp_path / "plan", world)


def test_allocated_run_is_refused_before_provider_access(tmp_path, monkeypatch):
    shutil.copytree(REPO / "tests/fixtures/worlds/habitat/1.0.0",
                    tmp_path / "worlds/habitat/1.0.0")
    (tmp_path / "runs/s012-001").mkdir(parents=True)

    def forbidden(*args, **kwargs):
        pytest.fail("provider must not be consulted after a run was allocated")

    monkeypatch.setattr(readiness.providers, "launch_check", forbidden)
    with pytest.raises(ValueError, match="already has an allocated run"):
        readiness.check(tmp_path)


def test_preflight_checks_local_state_without_reserving_a_request(tmp_path, monkeypatch):
    shutil.copytree(REPO / "tests/fixtures/worlds/habitat/1.0.0",
                    tmp_path / "worlds/habitat/1.0.0")
    frozen = json.loads((PLAN / "freeze.json").read_text())
    profile = {"limits": {"per_run": 180, "per_day": 500, "total": 500}}
    monkeypatch.setattr(readiness.providers, "launch_check",
                        lambda *a: {"profile": profile})
    original = readiness.verify_plan

    def prepared(directory, world):
        config, _ = original(directory, world)
        return config, {**frozen, "provider_profile_sha256": readiness.digest(
            readiness.canonical(profile))}

    monkeypatch.setattr(readiness, "verify_plan", prepared)
    assert readiness.check(tmp_path)["experiment_id"] == "s012-001"
    assert readiness.providers.Ledger(tmp_path, "openai-chatgpt").entries() == []
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("problem", ["profile", "capacity"])
def test_changed_profile_or_insufficient_capacity_refuses_launch(tmp_path, monkeypatch, problem):
    shutil.copytree(REPO / "tests/fixtures/worlds/habitat/1.0.0",
                    tmp_path / "worlds/habitat/1.0.0")
    profile = {"limits": {"per_run": 180, "per_day": 100, "total": 100}}
    monkeypatch.setattr(readiness.providers, "launch_check",
                        lambda *a: {"profile": profile})
    original = readiness.verify_plan

    def prepared(directory, world):
        config, frozen = original(directory, world)
        pin = readiness.digest(readiness.canonical(profile)) if problem == "capacity" else "old"
        return config, {**frozen, "provider_profile_sha256": pin}

    monkeypatch.setattr(readiness, "verify_plan", prepared)
    message = "profile changed" if problem == "profile" else "capacity"
    with pytest.raises(ValueError, match=message):
        readiness.check(tmp_path)
    assert readiness.providers.Ledger(tmp_path, "openai-chatgpt").entries() == []
    assert not (tmp_path / "runs").exists()
