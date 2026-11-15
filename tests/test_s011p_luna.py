"""S011p: the frozen Luna smoke configuration, run end to end against a fake provider.

The configuration file under ``experiments/s011p/`` is the one a live NOT_SCORED
check would launch. Here it runs through the real supervisor, broker and adapter
against a local fake of the Responses endpoint, so the outgoing payload can be
inspected: every request must name ``gpt-5.6-luna`` and carry
``reasoning.effort=low`` (owner requirement; the provider default is medium).

Scripted success is integration evidence only, never evidence of model behaviour.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import pytest

from conftest import REPO, events_of
from open_lab_simulation import chatgpt, chatgpt_auth, credentials, providers
from open_lab_simulation.archive import verify_archive
from open_lab_simulation.contracts import canonical_json, sha256_text
from open_lab_simulation.supervisor import run_experiment
from test_hosted_compatibility import ANSWER, DESCRIBE, INVOKE, SEARCH
from test_providers import Upstream

SMOKE = REPO / "experiments" / "s011p" / "smoke.json"
WORLD = REPO / "tests" / "fixtures" / "worlds" / "minimal" / "0.8.0"
PROVIDER = chatgpt_auth.PROVIDER_ID
ACCESS = "TEST-s011p-access-token-0123456789"


def smoke_config() -> dict:
    return json.loads(SMOKE.read_text(encoding="utf-8"))


def test_smoke_config_pins_luna_low_effort_and_its_catalog():
    config = smoke_config()
    extensions = config["extensions"]
    assert extensions["hosted_inference"] == {"reasoning_effort": "low"}
    catalog = extensions["model_catalog"]
    assert sha256_text(canonical_json(catalog["document"])) == catalog["sha256"]
    (model,) = catalog["document"]["models"]
    assert model["provider_ref"] == "gpt-5.6-luna"
    assert model["id"] == config["orchestrator"]["model_id"]
    assert config["permissions"]["models"] == [model["id"]]
    # Client guard equals the provider-reported context_window, never the larger maximum.
    assert model["context_window_tokens"] == 272_000
    assert extensions["runtime_policy"]["context_guard_tokens"] == 272_000
    assert "not-scored" in config["tags"] and "benchmark" not in extensions


@pytest.fixture
def luna_run(tmp_path, monkeypatch):
    monkeypatch.setattr(credentials, "credentials_dir", lambda: tmp_path / "credentials")
    root = tmp_path / "data"
    shutil.copytree(WORLD, root / "worlds/minimal/0.8.0")
    upstreams = []

    def run(script, *, per_run=8):
        upstream = Upstream(script)
        upstreams.append(upstream)
        monkeypatch.setattr(chatgpt, "ENDPOINT", upstream.url)
        adapter = chatgpt.ChatGPTAdapter()
        adapter.host = "127.0.0.1"
        monkeypatch.setitem(providers.ADAPTERS, PROVIDER, adapter)
        credentials.store(PROVIDER, json.dumps({
            "client_id": "oaiapp_s011p", "access_token": ACCESS, "refresh_token": "TEST-r",
            "id_token": None, "scopes": sorted(chatgpt_auth.SCOPES.split()),
            "expires_at": time.time() + 3600}, separators=(",", ":")))
        providers.write_profile(root, PROVIDER, enabled=True, per_run=per_run, per_day=8,
                                total=8, max_attempts_per_step=1)
        # The frozen file unchanged except a shorter quiet-idle deadline, so the
        # suite does not wait out the live check's 180 s.
        config = smoke_config()
        config["limits"]["wall_clock_s"] = 15
        path = tmp_path / "smoke.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        return run_experiment(Path(path), root), upstream

    yield run
    for upstream in upstreams:
        upstream.close()


def test_every_luna_request_sends_low_effort_and_the_invoke_reaches_math(luna_run):
    result, upstream = luna_run([SEARCH, DESCRIBE, INVOKE, ANSWER])
    bodies = [r["body"] for r in upstream.requests]
    assert len(bodies) == 4
    for body in bodies:
        assert body["model"] == "gpt-5.6-luna"
        assert body["reasoning"] == {"effort": "low"}
        assert not {"temperature", "max_output_tokens", "seed", "top_p"} & set(body)
    events = events_of(result.archive)
    results = [e["payload"] for e in events if e["type"] == "tool.result"]
    assert results[-1]["status"] == "ok" and results[-1]["result"]["value"] == 42
    assert verify_archive(result.archive) == []
    frozen = json.loads((result.archive / "experiment.json").read_text(encoding="utf-8"))
    assert '"reasoning_effort": "low"' in json.dumps(frozen)


def test_the_smoke_cannot_exceed_its_request_ceiling(luna_run):
    result, upstream = luna_run([SEARCH] * 8, per_run=3)
    assert len(upstream.requests) == 3
    messages = [e["payload"]["message"] for e in events_of(result.archive) if e["type"] == "error"]
    assert any("laboratory_budget_exhausted" in m for m in messages)
