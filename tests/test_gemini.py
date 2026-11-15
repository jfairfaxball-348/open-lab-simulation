"""Gemini adapter translation and a full run against a fake Gemini server (S014c).

No request reaches Google here. The live check is a separate, approved manual
step recorded in the Builder's docs/experiments/.
"""

from __future__ import annotations

import json
import shutil

import pytest

from conftest import FIXTURE_WORLD, add_test_capabilities, events_of
from open_lab_simulation import credentials, gemini, providers
from open_lab_simulation.archive import verify_archive
from open_lab_simulation.contracts import canonical_json, sha256_text
from open_lab_simulation.model import tool_definitions
from open_lab_simulation.registry import META_TOOLS
from open_lab_simulation.supervisor import run_experiment
from test_providers import SECRET, Upstream

THOUGHT = "PRIVATE-REASONING-should-never-be-recorded"
SIGNATURE = "c2lnbmF0dXJlLWJ5dGVz"


def chat(messages):
    tools, _ = tool_definitions(META_TOOLS)
    return {"messages": messages, "tools": tools}


def call_reply(name="capabilities_search", args=None, ident=None):
    call = {"name": name, "args": args or {"query": "math"}}
    if ident:
        call["id"] = ident
    return {"candidates": [{"content": {"role": "model", "parts": [
        {"text": THOUGHT, "thought": True},
        {"functionCall": call, "thoughtSignature": SIGNATURE}]},
        "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 120, "candidatesTokenCount": 9,
                          "thoughtsTokenCount": 40, "totalTokenCount": 169},
        "modelVersion": "gemma-4-26b-a4b-it-test"}


TEXT = {"candidates": [{"content": {"role": "model", "parts": [{"text": "All done."}]},
                        "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 200, "candidatesTokenCount": 3}}


def test_request_keeps_schemas_and_maps_settings():
    adapter = gemini.GeminiAdapter()
    upstream = adapter.request(
        chat([{"role": "system", "content": "preamble"}, {"role": "user", "content": "hi"}]),
        {"model": "gemma-4-26b-a4b-it", "temperature": 0.7, "seed": 41,
         "max_output_tokens": 2048},
        {"inference": {"thinking_level": "minimal"}})
    assert upstream.url.endswith("/models/gemma-4-26b-a4b-it:generateContent")
    body = upstream.body
    assert body["systemInstruction"] == {"parts": [{"text": "preamble"}]}
    assert body["contents"] == [{"role": "user", "parts": [{"text": "hi"}]}]
    tools, _ = tool_definitions(META_TOOLS)
    sent = body["tools"][0]["functionDeclarations"]
    assert [d["parametersJsonSchema"] for d in sent] == [t["function"]["parameters"] for t in tools]
    assert body["generationConfig"] == {"temperature": 0.7, "seed": 41, "maxOutputTokens": 2048,
                                        "thinkingConfig": {"thinkingLevel": "minimal"}}
    assert adapter.auth_headers("k") == {"x-goog-api-key": "k"}
    with pytest.raises(ValueError):
        adapter.request(chat([]), {"model": "../x"}, {})


def test_response_hides_thoughts_and_history_resends_original_parts():
    adapter = gemini.GeminiAdapter()
    outcome = adapter.response(200, call_reply(), {}, {})
    message = outcome.reply["choices"][0]["message"]
    assert message["content"] == "" and THOUGHT not in json.dumps(outcome.reply)
    (call,) = message["tool_calls"]
    assert call["id"] == "gemini-call-1" and json.loads(call["function"]["arguments"]) == {
        "query": "math"}
    assert outcome.usage == {"prompt_tokens": 120, "completion_tokens": 9,
                             "reasoning_tokens": 40, "total_tokens": 169}
    assert outcome.model_version == "gemma-4-26b-a4b-it-test"
    history = [{"role": "user", "content": "hi"}, message,
               {"role": "tool", "tool_call_id": "gemini-call-1", "content": '{"matches": []}'},
               {"role": "tool", "tool_call_id": "gemini-call-1", "content": "plain"},
               {"role": "user", "content": "continue"}]
    body = adapter.request(chat(history), {"model": "gemma-4-26b-a4b-it"},
                           {**outcome.state, "inference": {}}).body
    model_turn = body["contents"][1]
    assert model_turn["role"] == "model" and THOUGHT not in json.dumps(body)
    assert model_turn["parts"] == [{"functionCall": {"name": "capabilities_search",
                                                     "args": {"query": "math"},
                                                     "id": "gemini-call-1"},
                                    "thoughtSignature": SIGNATURE}]
    # Tool results and the continuation merge into one user turn, ids and names kept.
    assert body["contents"][2] == {"role": "user", "parts": [
        {"functionResponse": {"name": "capabilities_search", "id": "gemini-call-1",
                              "response": {"matches": []}}},
        {"functionResponse": {"name": "capabilities_search", "id": "gemini-call-1",
                              "response": {"content": "plain"}}},
        {"text": "continue"}]}


def test_history_without_cached_original_is_rebuilt_faithfully():
    adapter = gemini.GeminiAdapter()
    message = {"role": "assistant", "content": "thinking aloud", "tool_calls": [
        {"id": "x1", "type": "function",
         "function": {"name": "capabilities_describe", "arguments": '{"id": "math.evaluate"}'}}]}
    body = adapter.request(chat([message]), {"model": "gemma-4-26b-a4b-it"}, {}).body
    assert body["contents"] == [{"role": "model", "parts": [
        {"text": "thinking aloud"},
        {"functionCall": {"name": "capabilities_describe", "args": {"id": "math.evaluate"},
                          "id": "x1"}}]}]


@pytest.mark.parametrize("status, payload, kind, code_part", [
    (429, {"error": {"status": "RESOURCE_EXHAUSTED", "details": [{"retryDelay": "7s"}]}},
     "retry", "RESOURCE_EXHAUSTED"),
    (429, {"error": {"status": "RESOURCE_EXHAUSTED", "details": [
        {"violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}]}},
     "terminal", "daily_quota"),
    (503, {"error": {"status": "UNAVAILABLE"}}, "retry", "UNAVAILABLE"),
    (400, {"error": {"status": "INVALID_ARGUMENT"}}, "terminal", "INVALID_ARGUMENT"),
    (403, None, "terminal", "http_403"),
])
def test_errors_are_classified(status, payload, kind, code_part):
    outcome = gemini.GeminiAdapter().response(status, payload, {}, {})
    assert outcome.kind == kind and code_part in outcome.code
    if code_part == "RESOURCE_EXHAUSTED" and kind == "retry":
        assert outcome.retry_after_s == 7


def test_settings_are_validated_and_reported():
    adapter = gemini.GeminiAdapter()
    support = adapter.settings_support({"temperature": 0.7, "seed": 1},
                                       {"thinking_level": "high"})
    assert "thinkingLevel=high" in support["thinking_level"]
    assert "effect not attested" in support["seed"]
    for bad in ({"thinking_level": "max"}, {"top_k": 3}):
        with pytest.raises(ValueError):
            adapter.settings_support({}, bad)


def test_full_run_against_a_fake_gemini_server(tmp_path, monkeypatch, make_config):
    monkeypatch.setattr(credentials, "credentials_dir", lambda: tmp_path / "credentials")
    upstream = Upstream([(200, call_reply()), (200, TEXT)])
    monkeypatch.setattr(gemini, "ENDPOINT", upstream.url + "/{model}")
    adapter = gemini.GeminiAdapter()
    adapter.host = "127.0.0.1"
    monkeypatch.setitem(providers.ADAPTERS, "google-gemini", adapter)
    root = tmp_path / "data"
    shutil.copytree(FIXTURE_WORLD, root / "worlds/minimal/0.6.0")
    add_test_capabilities(root / "worlds/minimal/0.6.0")
    credentials.store("google-gemini", SECRET)
    providers.write_profile(root, "google-gemini", enabled=True, per_run=5, per_day=10, total=10)
    document = {
        "schema": "open-lab/model-provider", "schema_version": "1.0.0",
        "provider": {"id": "google-gemini", "name": "Google Gemini API", "kind": "other",
                     "locality": "external",
                     "api": {"protocol": "openai_chat_completions",
                             "base_url": "https://generativelanguage.googleapis.com/v1beta"}},
        "models": [{"id": "orchestrator-gemma-4-26b", "provider_ref": "gemma-4-26b-a4b-it",
                    "roles": ["orchestrator", "text_generation"],
                    "modalities": {"input": ["text"], "output": ["text"]},
                    "context_window_tokens": 4096, "status": "available"}],
    }
    config = make_config(
        [], limits={"wall_clock_s": 60, "max_steps": 2},
        orchestrator={"provider_id": "google-gemini", "model_id": "orchestrator-gemma-4-26b",
                      "sampling": {"temperature": 0.7, "max_output_tokens": 2048}},
        permissions={"filesystem": "workspace_write", "network": "loopback",
                     "models": ["orchestrator-gemma-4-26b"]},
        extensions={"model_catalog": {"document": document,
                                      "sha256": sha256_text(canonical_json(document))},
                    "hosted_inference": {"thinking_level": "minimal"}})
    try:
        result = run_experiment(config, root)
    finally:
        upstream.close()
    events = events_of(result.archive)
    results = [e["payload"] for e in events if e["type"] == "tool.result"]
    assert results and results[0]["status"] == "ok"
    assert len(upstream.requests) == 2
    assert all(r["headers"]["X-Goog-Api-Key"] == SECRET for r in upstream.requests)
    second = upstream.requests[1]["body"]
    assert second["contents"][1]["parts"][0]["thoughtSignature"] == SIGNATURE
    assert second["contents"][2]["parts"][0]["functionResponse"]["id"] == "gemini-call-1"
    assert THOUGHT not in json.dumps(upstream.requests[1]["body"])
    for path in result.archive.rglob("*"):
        if path.is_file():
            text = path.read_bytes().decode("utf-8", "replace")
            assert THOUGHT not in text and SECRET not in text, path.name
    model = json.loads((result.archive / "model.json").read_text())
    assert model["settings"]["sampling_effective"]["thinking_level"].endswith("minimal")
    assert verify_archive(result.archive) == []
