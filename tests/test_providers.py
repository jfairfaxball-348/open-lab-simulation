"""Hosted provider broker, budgets, refusals and credential isolation (S014b, ADR-0016).

No real provider is contacted. A fake adapter talks to a fake upstream server on
loopback; the code paths under test (profiles, ledger, broker, supervisor wiring,
redaction) are the ones every real hosted adapter uses.
"""

from __future__ import annotations

import json
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from conftest import FIXTURE_WORLD, add_test_capabilities, events_of
from open_lab_simulation import cli, credentials, providers
from open_lab_simulation.archive import verify_archive
from open_lab_simulation.contracts import canonical_json, jsonl_lines, sha256_text
from open_lab_simulation.supervisor import LaunchError, run_experiment

SECRET = "TEST-provider-credential-0123456789abcdef"
PROVIDER = "fake-hosted"


def search_call(query="math"):
    return {"choices": [{"message": {"role": "assistant", "content": "", "tool_calls": [{
        "id": "c1", "type": "function",
        "function": {"name": "capabilities_search", "arguments": json.dumps({"query": query})},
    }]}}]}


TEXT = {"choices": [{"message": {"role": "assistant", "content": "done"}}]}


class Upstream:
    """A scripted stand-in for a provider API, recording what reached it."""

    def __init__(self, script):
        self.script = list(script)
        self.requests = []
        upstream = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                upstream.requests.append({"headers": dict(self.headers), "body": body})
                step = upstream.script.pop(0) if upstream.script else (200, TEXT)
                if isinstance(step, tuple) and step[0] == "slow":  # ("slow", seconds, step)
                    time.sleep(step[1])
                    step = step[2]
                if step == "drop":
                    self.close_connection = True
                    self.wfile.flush()
                    self.connection.close()
                    return
                status, payload = step
                streamed = isinstance(payload, bytes)  # a server-sent event stream
                data = payload if streamed else json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type",
                                 "text/event-stream" if streamed else "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/chat"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class FakeAdapter:
    id = "fake-chat"
    version = "1.0.0"
    host = "127.0.0.1"

    def __init__(self, upstream):
        self.upstream = upstream

    def request(self, chat, settings, state):
        body = {"model": settings["model"], "messages": chat["messages"], "tools": chat["tools"]}
        return providers.Upstream(self.upstream.url, body)

    def auth_headers(self, secret):
        return {"X-Fake-Key": secret}

    def response(self, status, payload, headers, state):
        if status == 200:
            return providers.Outcome("ok", "ok", reply=payload,
                                     usage={"prompt_tokens": 10, "completion_tokens": 2})
        if status in (429, 503):
            return providers.Outcome("retry", f"http_{status}", retry_after_s=0.01)
        return providers.Outcome("terminal", f"http_{status}")

    def settings_support(self, sampling, inference):
        return {name: "honoured" for name in sampling}


@pytest.fixture
def hosted(tmp_path, monkeypatch):
    """A data root, a stored test credential, an enabled profile and a fake upstream."""
    monkeypatch.setattr(credentials, "credentials_dir", lambda: tmp_path / "credentials")
    root = tmp_path / "data"
    release = root / "worlds/minimal/0.6.0"
    shutil.copytree(FIXTURE_WORLD, release)
    add_test_capabilities(release)
    upstreams = []

    def make(script=(), **limits):
        upstream = Upstream(script)
        upstreams.append(upstream)
        monkeypatch.setitem(providers.ADAPTERS, PROVIDER, FakeAdapter(upstream))
        credentials.store(PROVIDER, SECRET)
        providers.write_profile(root, PROVIDER, enabled=True,
                                **{"per_run": 10, "per_day": 100, "total": 1000, **limits})
        return upstream

    yield root, make
    for upstream in upstreams:
        upstream.close()


def catalog():
    document = {
        "schema": "open-lab/model-provider", "schema_version": "1.0.0",
        "provider": {"id": PROVIDER, "name": "Fake hosted provider", "kind": "other",
                     "locality": "external",
                     "api": {"protocol": "openai_chat_completions",
                             "base_url": "https://provider.invalid/v1"}},
        "models": [{"id": "hosted-fake", "provider_ref": "fake-model-1",
                    "roles": ["orchestrator", "text_generation"],
                    "modalities": {"input": ["text"], "output": ["text"]},
                    "context_window_tokens": 4096, "status": "available"}],
    }
    return {"document": document, "sha256": sha256_text(canonical_json(document))}


@pytest.fixture
def hosted_config(make_config):
    def make(max_steps=3, **changes):
        extensions = {"model_catalog": catalog(), **changes.pop("extensions", {})}
        return make_config([], limits={"wall_clock_s": 60, "max_steps": max_steps},
                           orchestrator={"provider_id": PROVIDER, "model_id": "hosted-fake",
                                         "sampling": {"temperature": 0.7}},
                           permissions={"filesystem": "workspace_write", "network": "loopback",
                                        "models": ["hosted-fake"]},
                           extensions=extensions, **changes)
    return make


def ledger(root):
    return providers.Ledger(root, PROVIDER).entries()


def files_text(directory: Path) -> str:
    return "".join(p.read_bytes().decode("utf-8", "replace")
                   for p in directory.rglob("*") if p.is_file())


def test_hosted_run_goes_through_the_broker_and_never_exposes_the_credential(
        hosted, hosted_config, capsys):
    root, make = hosted
    upstream = make([(200, search_call()), (200, TEXT)])
    result = run_experiment(hosted_config(max_steps=3), root)
    assert (result.state, result.reason) == ("terminated", "max_steps"), result.reason
    events = events_of(result.archive)
    results = [e["payload"] for e in events if e["type"] == "tool.result"]
    assert results[0]["status"] == "ok" and results[0]["result"]["matches"]
    assert sum(e["type"] == "model.call" for e in events) == 3
    # Exactly one reserved and completed provider attempt per model step.
    reserves = [e for e in ledger(root) if e["kind"] == "reserve"]
    outcomes = [e for e in ledger(root) if e["kind"] == "outcome"]
    assert len(reserves) == len(outcomes) == len(upstream.requests) == 3
    assert {e["run_id"] for e in reserves} == {result.run_id}
    # The provider got the credential; nothing the run produced contains it.
    assert all(r["headers"]["X-Fake-Key"] == SECRET for r in upstream.requests)
    assert SECRET not in files_text(result.archive)
    assert SECRET not in files_text(result.workspace)
    assert SECRET not in (root / "providers" / PROVIDER / "ledger.jsonl").read_text()
    assert SECRET not in capsys.readouterr().err
    model = json.loads((result.archive / "model.json").read_text())
    assert "auth_token" not in json.dumps(model) and model["base_url"] is None
    assert model["hosted"]["adapter"] == {"id": "fake-chat", "version": "1.0.0",
                                          "host": "127.0.0.1"}
    assert model["identity"]["digest"] is None
    mirrored = jsonl_lines((result.archive / "provider-ledger.jsonl").read_text(encoding="utf-8"))
    assert sum('"kind":"request"' in line for line in mirrored) == 3
    assert verify_archive(result.archive) == []


@pytest.mark.parametrize("setup, expected", [
    ("no-profile", "not configured"),
    ("disabled", "disabled"),
    ("no-credential", "no credential"),
    ("recovery", "recovery is not supported"),
])
def test_unapproved_hosted_launches_are_refused(hosted, hosted_config, setup, expected):
    root, make = hosted
    upstream = make()
    changes = {}
    if setup == "no-profile":
        (root / "providers" / PROVIDER / "profile.json").unlink()
    elif setup == "disabled":
        providers.write_profile(root, PROVIDER, enabled=False)
    elif setup == "no-credential":
        credentials.delete(PROVIDER)
    else:
        changes.update(extensions={"recovery": {"max_restarts": 1}}, checkpointing={})
    with pytest.raises(LaunchError, match=expected):
        run_experiment(hosted_config(**changes), root)
    assert not (root / "runs").exists() and upstream.requests == []


def test_per_run_budget_stops_the_run_without_another_request(hosted, hosted_config):
    root, make = hosted
    upstream = make([(200, search_call())] * 5, per_run=2)
    result = run_experiment(hosted_config(max_steps=5), root)
    assert len(upstream.requests) == 2
    events = events_of(result.archive)
    errors = [e["payload"]["message"] for e in events if e["type"] == "error"]
    assert any("laboratory_budget_exhausted" in m for m in errors), errors
    actions = [e["payload"] for e in events if e["type"] == "supervisor.action"]
    assert any(a.get("extensions", {}).get("limit") == "provider_per_run" for a in actions)
    assert verify_archive(result.archive) == []


def test_retries_are_reserved_and_terminal_errors_stop_the_provider(hosted, hosted_config):
    root, make = hosted
    upstream = make([(429, {}), (200, search_call()), (400, {}), (200, TEXT)])
    result = run_experiment(hosted_config(max_steps=4), root)
    # Step 1: 429 then success (two reservations). Step 2: terminal 400. No further sends.
    assert len(upstream.requests) == 3
    reserves = [e for e in ledger(root) if e["kind"] == "reserve"]
    assert [e["attempt"] for e in reserves] == [1, 2, 1]
    errors = [e["payload"]["message"] for e in events_of(result.archive) if e["type"] == "error"]
    assert any("laboratory_provider_terminal" in m for m in errors), errors


def test_a_dropped_connection_is_counted_as_spent(hosted, hosted_config):
    root, make = hosted
    upstream = make(["drop", (200, TEXT)], per_run=10)
    run_experiment(hosted_config(max_steps=1), root)
    reserves = [e for e in ledger(root) if e["kind"] == "reserve"]
    outcomes = [e for e in ledger(root) if e["kind"] == "outcome"]
    assert len(reserves) == 2 and len(upstream.requests) == 2
    assert outcomes[0]["result"] == "retry" and outcomes[0]["code"].startswith("transport_")


def test_daily_budget_is_shared_across_runs(hosted, hosted_config):
    root, make = hosted
    make([(200, TEXT)] * 4, per_day=2)
    run_experiment(hosted_config(max_steps=2), root)
    with pytest.raises(LaunchError, match="budget 'per_day' is already exhausted"):
        run_experiment(hosted_config(max_steps=2), root)


def test_broker_requires_the_run_token(hosted, tmp_path):
    root, make = hosted
    upstream = make([(200, TEXT)])
    archive = tmp_path / "archive"
    archive.mkdir()
    plan = providers.Plan(PROVIDER, "fake-model-1", {}, {}, providers.load_profile(root, PROVIDER),
                          "run-x", None, archive)
    broker = providers.Broker(root, plan).start()
    try:
        body = json.dumps({"messages": [], "tools": []}).encode()
        for headers in ({}, {"Authorization": "Bearer wrong"}):
            request = urllib.request.Request(f"{broker.url}/chat/completions", data=body,
                                             headers=headers, method="POST")
            with pytest.raises(urllib.error.HTTPError) as caught:
                urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request)
            assert caught.value.code == 401
    finally:
        broker.stop()
    assert upstream.requests == [] and ledger(root) == []


def test_profiles_validate_and_default_to_disabled(tmp_path):
    profile = providers.write_profile(tmp_path, "google-gemini", per_run=3)
    assert profile["enabled"] is False and profile["limits"] == {
        "per_run": 3, "per_day": 0, "total": 0}
    with pytest.raises(providers.ProviderError):
        providers.validate_profile({**profile, "limits": {"per_run": -1}}, "google-gemini")
    with pytest.raises(providers.ProviderError):
        providers.write_profile(tmp_path, "../escape", enabled=True)


@pytest.mark.skipif(sys.platform != "win32", reason="DPAPI credential store is Windows-only")
def test_credentials_are_encrypted_at_rest(tmp_path, monkeypatch):
    monkeypatch.setattr(credentials, "credentials_dir", lambda: tmp_path)
    credentials.store("google-gemini", SECRET)
    stored = (tmp_path / "google-gemini.dpapi").read_bytes()
    assert SECRET.encode() not in stored
    assert credentials.load("google-gemini") == SECRET
    with pytest.raises(credentials.CredentialError):
        credentials.store("google-gemini", "has space")
    assert credentials.delete("google-gemini") and not credentials.exists("google-gemini")
    with pytest.raises(credentials.CredentialError, match="no credential") as caught:
        credentials.load("google-gemini")
    assert SECRET not in str(caught.value)


def test_cli_reads_the_credential_without_echoing_it(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(credentials, "credentials_dir", lambda: tmp_path / "credentials")
    monkeypatch.setattr(cli.getpass, "getpass", lambda prompt: SECRET)
    root = tmp_path / "data"
    assert cli.main(["provider", "set-key", "google-gemini", "--data-root", str(root)]) == 0
    assert cli.main(["provider", "configure", "google-gemini", "--enable", "--per-run", "4",
                     "--per-day", "8", "--total", "20", "--data-root", str(root)]) == 0
    assert cli.main(["provider", "status", "google-gemini", "--data-root", str(root)]) == 0
    assert cli.main(["provider", "set-key", "--data-root", str(root)]) == 2
    output = capsys.readouterr()
    assert SECRET not in output.out + output.err
    assert credentials.load("google-gemini") == SECRET


def _broker(root, tmp_path, name, **profile):
    archive = tmp_path / name
    archive.mkdir()
    loaded = {**providers.load_profile(root, PROVIDER), **profile}
    plan = providers.Plan(PROVIDER, "fake-model-1", {}, {}, loaded, f"run-{name}", None, archive)
    return providers.Broker(root, plan).start(), archive


def _post(broker):
    body = json.dumps({"messages": [], "tools": []}).encode()
    request = urllib.request.Request(f"{broker.url}/chat/completions", data=body, method="POST",
                                     headers={"Authorization": f"Bearer {broker.token}"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_a_new_run_waits_for_a_stopped_runs_in_flight_request(hosted, tmp_path):
    """S014i pilot regression: a deadline stop left a request in flight, and the next
    cell's broker was refused the provider lock and mislabelled it a bad request."""
    root, make = hosted
    upstream = make([("slow", 1.5, (200, TEXT)), (200, TEXT)])
    first, first_archive = _broker(root, tmp_path, "first")
    second, second_archive = _broker(root, tmp_path, "second")
    try:
        caller = threading.Thread(target=_post, args=(first,), daemon=True)
        caller.start()
        time.sleep(0.5)  # the first request is now in flight upstream
        first.stop()  # the run ends at its deadline
        status, body = _post(second)  # the next cell starts at once
        caller.join(5)
    finally:
        second.stop()
    assert status == 200 and body["choices"][0]["message"]["content"] == "done"
    assert len(upstream.requests) == 2
    entries = ledger(root)
    late = next(e for e in entries if e["kind"] == "outcome" and e.get("after_run_stop"))
    assert late["result"] == "ok"
    # The sealed first archive keeps its reservation but receives nothing afterwards.
    mirrored = [json.loads(line) for line in
                jsonl_lines((first_archive / "provider-ledger.jsonl").read_text(encoding="utf-8"))]
    assert [e["kind"] for e in mirrored] == ["reserve", "request"]
    assert entries.index(late) < next(i for i, e in enumerate(entries)
                                      if e["kind"] == "reserve" and e["run_id"] == "run-second")


def test_a_held_provider_lock_is_reported_as_busy(hosted, tmp_path, monkeypatch):
    root, make = hosted
    upstream = make([(200, TEXT)])
    monkeypatch.setattr(providers, "LOCK_GRACE_S", 0)
    broker, _ = _broker(root, tmp_path, "blocked", request_timeout_s=0.5)
    try:
        with providers.archive_lock(providers.provider_dir(root, PROVIDER), recover_stale=True):
            status, body = _post(broker)
    finally:
        broker.stop()
    assert status == 503 and body["error"]["code"] == "laboratory_provider_busy"
    assert upstream.requests == [] and ledger(root) == []


def test_min_interval_spaces_every_send_including_retries(hosted, tmp_path):
    """S011p pacing: the broker's min_interval_s is the registered request cadence.
    Every reserved attempt, retries included, starts at least that long after the last."""
    root, make = hosted
    make([(503, {}), (200, TEXT), (200, TEXT)])
    broker, _ = _broker(root, tmp_path, "paced", min_interval_s=0.6, max_attempts_per_step=2)
    try:
        assert _post(broker)[0] == 200  # one 503 retry, then ok
        assert _post(broker)[0] == 200
    finally:
        broker.stop()
    sends = [e["time"] for e in ledger(root) if e["kind"] == "reserve"]
    assert len(sends) == 3
    # Pacing uses the monotonic clock; ledger times are wall-clock stamps whose Windows
    # resolution is coarse, hence the small tolerance.
    assert all(later - earlier >= 0.55 for earlier, later in zip(sends, sends[1:], strict=False))
