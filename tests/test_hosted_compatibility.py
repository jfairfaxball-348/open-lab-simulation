"""S014g hosted compatibility suite: does a provider preserve Laboratory tool semantics?

Each case is a full supervised run (supervisor, runtime, broker, adapter) against a
fake provider server, so the code paths are the ones a live run uses. Only
providers that pass every case may enter the matched campaign. The suite is
parameterised by provider; after the owner's 2026-11-15 decisions (Gemma
scrapped, Anthropic excluded) it covers ChatGPT plan usage alone.

Scripted success here is integration evidence only, never evidence of autonomy.
"""

from __future__ import annotations

import json
import shutil
import time

import pytest

from conftest import FIXTURE_WORLD, add_test_capabilities, events_of
from open_lab_simulation import chatgpt, chatgpt_auth, credentials, providers
from open_lab_simulation.archive import verify_archive
from open_lab_simulation.contracts import canonical_json, sha256_text
from open_lab_simulation.supervisor import LaunchError, resume_experiment, run_experiment
from test_providers import Upstream

ACCESS = "TEST-compat-access-token-0123456789"
REFRESH = "TEST-compat-refresh-token-abcdef"
REASONING = "PRIVATE-compat-reasoning-never-recorded"
PROVIDER = chatgpt_auth.PROVIDER_ID
MODEL = "orchestrator-chatgpt-compat"
GUARD = 500_000  # the OD-13 registered client guard for the ChatGPT arm


# --- the provider's wire format --------------------------------------------------------


def sse(*events) -> bytes:
    return b"".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n".encode() for e in events)


def completed(*output):
    return (200, sse({"type": "response.created", "response": {}}, {
        "type": "response.completed", "response": {
            "model": "gpt-compat-1", "output": [
                {"type": "reasoning", "id": "rs_x",
                 "summary": [{"type": "summary_text", "text": REASONING}]}, *output],
            "usage": {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110}}}))


def call(name, args, ident):
    return {"type": "function_call", "id": f"fc_{ident}", "call_id": ident,
            "namespace": "laboratory", "name": name, "arguments": json.dumps(args),
            "status": "completed"}


def text(value):
    return {"type": "message", "role": "assistant",
            "content": [{"type": "output_text", "text": value}]}


def failed(code):
    return (200, sse({"type": "response.failed", "response": {"error": {"code": code}}}))


def error(status, code):
    return (status, {"error": {"code": code, "message": "fake"}})


SEARCH = completed(call("capabilities_search", {"query": "arithmetic"}, "call_1"))
DESCRIBE = completed(call("capabilities_describe", {"id": "math.evaluate"}, "call_2"))
INVOKE = completed(call("capabilities_invoke",
                        {"id": "math.evaluate", "arguments": {"expression": "6*7"}}, "call_3"))
ANSWER = completed(text("math.evaluate returned 42."))


# --- one supervised run against the fake ----------------------------------------------


@pytest.fixture
def chatgpt_run(tmp_path, monkeypatch, make_config):
    monkeypatch.setattr(credentials, "credentials_dir", lambda: tmp_path / "credentials")
    root = tmp_path / "data"
    shutil.copytree(FIXTURE_WORLD, root / "worlds/minimal/0.6.0")
    add_test_capabilities(root / "worlds/minimal/0.6.0")
    upstreams = []

    def store_tokens(expires_in=3600):
        credentials.store(PROVIDER, json.dumps({
            "client_id": "oaiapp_compat", "access_token": ACCESS, "refresh_token": REFRESH,
            "id_token": None, "scopes": sorted(chatgpt_auth.SCOPES.split()),
            "expires_at": time.time() + expires_in}, separators=(",", ":")))

    def config(max_steps, **changes):
        document = {
            "schema": "open-lab/model-provider", "schema_version": "1.0.0",
            "provider": {"id": PROVIDER, "name": "OpenAI via ChatGPT plan usage",
                         "kind": "other", "locality": "external",
                         "api": {"protocol": "openai_chat_completions",
                                 "base_url": "https://api.openai.com/v1"}},
            "models": [{"id": MODEL, "provider_ref": "gpt-compat",
                        "roles": ["orchestrator", "text_generation"],
                        "modalities": {"input": ["text"], "output": ["text"]},
                        "context_window_tokens": GUARD, "status": "available"}],
        }
        extensions = {"model_catalog": {"document": document,
                                        "sha256": sha256_text(canonical_json(document))},
                      **changes.pop("extensions", {})}
        return make_config(
            [], limits={"wall_clock_s": 60, "max_steps": max_steps},
            orchestrator={"provider_id": PROVIDER, "model_id": MODEL,
                          "sampling": {"temperature": 0.7, "max_output_tokens": 2048,
                                       "seed": 41}},
            permissions={"filesystem": "workspace_write", "network": "loopback",
                         "models": [MODEL]},
            extensions=extensions, **changes)

    def run(script, *, max_steps, timeout=180, attempts=2, per_run=50, expires_in=3600,
            pause_after_steps=None, **changes):
        upstream = Upstream(script)
        upstreams.append(upstream)
        monkeypatch.setattr(chatgpt, "ENDPOINT", upstream.url)
        adapter = chatgpt.ChatGPTAdapter()
        adapter.host = "127.0.0.1"
        monkeypatch.setitem(providers.ADAPTERS, PROVIDER, adapter)
        store_tokens(expires_in)
        providers.write_profile(root, PROVIDER, enabled=True, per_run=per_run, per_day=500,
                                total=500, request_timeout_s=timeout,
                                max_attempts_per_step=attempts)
        result = run_experiment(config(max_steps, **changes), root,
                                pause_after_steps=pause_after_steps)
        return result, upstream

    run.root = root
    yield run
    for upstream in upstreams:
        upstream.close()


def errors_of(result):
    return [e["payload"]["message"] for e in events_of(result.archive) if e["type"] == "error"]


def outcomes(root):
    return [e for e in providers.Ledger(root, PROVIDER).entries() if e["kind"] == "outcome"]


def assert_clean(result):
    """Every case: the archive verifies and holds no token and no reasoning text."""
    assert verify_archive(result.archive) == []
    for path in result.archive.rglob("*"):
        if path.is_file():
            data = path.read_bytes().decode("utf-8", "replace")
            for secret in (ACCESS, REFRESH, REASONING):
                assert secret not in data, path.name


# --- the compatibility cases -----------------------------------------------------------


def test_text_reply(chatgpt_run):
    result, upstream = chatgpt_run([ANSWER], max_steps=1)
    outputs = [e["payload"] for e in events_of(result.archive) if e["type"] == "agent.output"]
    assert outputs[0]["text"] == "math.evaluate returned 42."
    assert "raw_tool_calls" not in outputs[0].get("extensions", {})
    assert len(upstream.requests) == 1
    assert_clean(result)


def test_structured_tool_call_reaches_a_real_capability(chatgpt_run):
    result, upstream = chatgpt_run([SEARCH, DESCRIBE, INVOKE, ANSWER], max_steps=4)
    events = events_of(result.archive)
    calls = [e["payload"] for e in events if e["type"] == "tool.call"]
    assert [c["capability_id"] for c in calls] == [
        "capabilities.search", "capabilities.describe", "capabilities.invoke"]
    assert calls[2]["arguments"] == {"id": "math.evaluate", "arguments": {"expression": "6*7"}}
    results = [e["payload"] for e in events if e["type"] == "tool.result"]
    assert [r["status"] for r in results] == ["ok", "ok", "ok"]
    assert results[2]["result"]["value"] == 42
    assert_clean(result)


def test_results_return_to_the_model_and_context_is_kept_whole(chatgpt_run):
    result, upstream = chatgpt_run([SEARCH, DESCRIBE, INVOKE, ANSWER], max_steps=4)
    bodies = [r["body"] for r in upstream.requests]
    assert len(bodies) == 4
    # Each request carries the whole history so far, in order, under the 500k guard.
    for earlier, later in zip(bodies, bodies[1:], strict=False):
        assert later["input"][:len(earlier["input"])] == earlier["input"]
    final = bodies[3]["input"]
    assert [i.get("type", i.get("role")) for i in final] == [
        "user", "function_call", "function_call_output", "function_call",
        "function_call_output", "function_call", "function_call_output"]
    assert [i["call_id"] for i in final if i.get("type") == "function_call_output"] == [
        "call_1", "call_2", "call_3"]
    invoke_output = json.loads(final[-1]["output"])
    assert "42" in json.dumps(invoke_output)
    # The documentation result reached the model untruncated.
    describe = next(e["payload"]["result"] for e in events_of(result.archive)
                    if e["type"] == "tool.result" and "documentation" in e["payload"]["result"])
    assert json.loads(final[4]["output"])["result"]["documentation"] == describe["documentation"]
    for body in bodies:
        assert body["store"] is False and body["stream"] is True
        assert "reasoning" not in [i.get("type") for i in body["input"]]
        assert not {"temperature", "max_output_tokens", "seed", "top_p"} & set(body)
    model_calls = [e["payload"] for e in events_of(result.archive) if e["type"] == "model.call"]
    assert not any(c.get("context_dropped_messages") or c.get("context_truncated_tool_results")
                   for c in model_calls)
    assert_clean(result)


def test_per_run_budget_is_enforced(chatgpt_run):
    result, upstream = chatgpt_run([SEARCH] * 4, max_steps=4, per_run=2)
    assert len(upstream.requests) == 2
    assert any("laboratory_budget_exhausted" in m for m in errors_of(result))
    assert_clean(result)


def test_rate_limit_is_retried_within_the_step(chatgpt_run):
    result, upstream = chatgpt_run([error(429, "rate_limit_exceeded"), ANSWER], max_steps=1)
    assert len(upstream.requests) == 2
    assert [o["result"] for o in outcomes(chatgpt_run.root)] == ["retry", "ok"]
    assert not errors_of(result)
    assert_clean(result)


@pytest.mark.parametrize("step, code", [
    (error(429, "subscription_sharing_usage_limit_exceeded"), "usage_limit_exceeded"),
    (failed("subscription_sharing_usage_limit_exceeded"), "usage_limit_exceeded"),
    (error(403, "subscription_sharing_user_not_eligible"), "user_not_eligible"),
    (error(400, "subscription_sharing_unsupported_capability"), "unsupported_capability"),
    (error(401, "subscription_sharing_invalid_user"), "invalid_user"),
])
def test_plan_refusals_stop_the_provider_without_retry(chatgpt_run, step, code):
    result, upstream = chatgpt_run([step, ANSWER], max_steps=2)
    assert len(upstream.requests) == 1
    assert [o["code"] for o in outcomes(chatgpt_run.root)][0].find(code) >= 0
    assert any("laboratory_provider_terminal" in m for m in errors_of(result))
    assert_clean(result)


@pytest.mark.parametrize("step", [
    error(503, "subscription_sharing_usage_unavailable"),
    (503, {"detail": "direct routing unavailable"}),
    (200, sse({"type": "response.created", "response": {}})),  # stream cut short
    "drop",
])
def test_temporary_failures_are_bounded_and_counted(chatgpt_run, step):
    result, upstream = chatgpt_run([step, step, ANSWER], max_steps=1)
    assert len(upstream.requests) == 2  # max_attempts_per_step, then the step ends
    assert [o["result"] for o in outcomes(chatgpt_run.root)] == ["retry", "retry"]
    assert any("laboratory_provider_unavailable" in m for m in errors_of(result))
    assert_clean(result)


def test_timeout_is_counted_as_spent_and_retried(chatgpt_run):
    result, upstream = chatgpt_run([("slow", 1.5, ANSWER), ANSWER], max_steps=1, timeout=0.5)
    first, second = outcomes(chatgpt_run.root)
    assert first["result"] == "retry" and first["code"].startswith("transport_")
    assert second["result"] == "ok"
    assert len([e for e in providers.Ledger(chatgpt_run.root, PROVIDER).entries()
                if e["kind"] == "reserve"]) == 2
    assert_clean(result)


def test_ended_session_stops_the_run_and_clears_tokens(chatgpt_run, monkeypatch):
    monkeypatch.setattr(chatgpt_auth, "_post_form",
                        lambda url, fields: (400, {"error": "invalid_grant"}))
    result, upstream = chatgpt_run([ANSWER], max_steps=1, expires_in=10)
    assert upstream.requests == []  # nothing is sent without a usable token
    assert any("laboratory_provider_terminal" in m for m in errors_of(result))
    assert not credentials.exists(PROVIDER)
    assert_clean(result)


def test_automatic_recovery_is_refused_for_hosted_runs(chatgpt_run):
    with pytest.raises(LaunchError, match="recovery is not supported"):
        chatgpt_run([ANSWER], max_steps=1, checkpointing={},
                    extensions={"recovery": {"max_restarts": 1}})


def test_a_paused_hosted_run_cannot_be_resumed(chatgpt_run):
    result, upstream = chatgpt_run([SEARCH, ANSWER], max_steps=3, pause_after_steps=1,
                                   checkpointing={"interval_s": 1, "keep_last": 4})
    summary = json.loads((result.archive / "summary.json").read_text(encoding="utf-8"))
    assert summary["final_state"] == "paused"
    sent = len(upstream.requests)
    with pytest.raises(LaunchError, match="attestable model digest"):
        resume_experiment(result.archive, chatgpt_run.root)
    assert len(upstream.requests) == sent  # refusal sends nothing
    assert_clean(result)
