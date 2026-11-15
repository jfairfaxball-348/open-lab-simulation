"""ChatGPT plan-usage sign-in, tokens and Responses translation, all against fakes (S014d).

No request reaches OpenAI here. The owner's live sign-in and smoke run are a
separate, approved manual step recorded in the Builder's docs/experiments/.
"""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
import shutil
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from conftest import FIXTURE_WORLD, add_test_capabilities, events_of
from open_lab_simulation import chatgpt, chatgpt_auth, credentials, providers
from open_lab_simulation.archive import verify_archive
from open_lab_simulation.contracts import canonical_json, sha256_text
from open_lab_simulation.model import tool_definitions
from open_lab_simulation.registry import META_TOOLS
from open_lab_simulation.supervisor import run_experiment
from test_providers import Upstream

ACCESS = "TEST-chatgpt-access-token-0123456789"
REFRESH = "TEST-chatgpt-refresh-token-abcdef"
REASONING = "PRIVATE-REASONING-summary-never-recorded"
ISSUED = "oaiapp_test_issued_client"


# --- an RSA test key and signed ID tokens (stdlib only) --------------------------------


def _prime(bits: int) -> int:
    while True:
        candidate = secrets.randbits(bits) | (1 << (bits - 1)) | 1
        d, r = candidate - 1, 0
        while d % 2 == 0:
            d, r = d // 2, r + 1
        for _ in range(24):
            x = pow(secrets.randbelow(candidate - 3) + 2, d, candidate)
            if x in (1, candidate - 1):
                continue
            for _ in range(r - 1):
                x = pow(x, 2, candidate)
                if x == candidate - 1:
                    break
            else:
                break
        else:
            return candidate


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


@pytest.fixture(scope="module")
def key():
    e = 65537
    while True:
        p, q = _prime(512), _prime(512)
        phi = (p - 1) * (q - 1)
        if p != q and phi % e:
            break
    n = p * q
    return {"n": n, "e": e, "d": pow(e, -1, phi),
            "jwk": {"kty": "RSA", "kid": "test-key", "alg": "RS256",
                    "n": _b64(n.to_bytes((n.bit_length() + 7) // 8, "big")),
                    "e": _b64(e.to_bytes(3, "big"))}}


def id_token(key, *, alg="RS256", **claims) -> str:
    body = {"iss": chatgpt_auth.ISSUER, "aud": ISSUED, "sub": "user-subject-1",
            "exp": int(time.time()) + 3600, "iat": int(time.time()), **claims}
    header = _b64(json.dumps({"alg": alg, "kid": "test-key"}).encode())
    payload = _b64(json.dumps(body).encode())
    size = (key["n"].bit_length() + 7) // 8
    digest = chatgpt_auth._SHA256_INFO + hashlib.sha256(f"{header}.{payload}".encode()).digest()
    encoded = b"\x00\x01" + b"\xff" * (size - len(digest) - 3) + b"\x00" + digest
    signature = pow(int.from_bytes(encoded, "big"), key["d"], key["n"]).to_bytes(size, "big")
    return f"{header}.{payload}.{_b64(signature)}"


def test_id_token_validation(key):
    keys = [key["jwk"]]
    claims = chatgpt_auth.validate_id_token(id_token(key, nonce="n1"), ISSUED, "n1", keys)
    assert claims["sub"] == "user-subject-1"
    good = id_token(key, nonce="n1")
    header, payload, signature = good.split(".")
    forged = _b64(json.dumps({**json.loads(base64.urlsafe_b64decode(payload + "==")),
                              "sub": "attacker"}).encode())
    for token, nonce, message in (
            (id_token(key, nonce="other"), "n1", "nonce"),
            (id_token(key, nonce="n1", aud="someone-else"), "n1", "audience"),
            (id_token(key, nonce="n1", exp=int(time.time()) - 600), "n1", "expired"),
            (id_token(key, nonce="n1", iss="https://evil.invalid"), "n1", "issuer"),
            (f"{header}.{forged}.{signature}", "n1", "signature"),
            (id_token(key, alg="none", nonce="n1"), "n1", "algorithm"),
            ("not-a-token", "n1", "malformed")):
        with pytest.raises(chatgpt_auth.SignInError, match=message):
            chatgpt_auth.validate_id_token(token, ISSUED, nonce, keys)


# --- a fake OpenAI authorization server ------------------------------------------------


class FakeAuth:
    def __init__(self, key, scope=chatgpt_auth.SCOPES, subject="user-subject-1"):
        self.key, self.scope, self.subject = key, scope, subject
        self.posts: list[tuple[str, dict[str, str]]] = []
        self.refresh_reply: tuple[int, dict] | None = None
        self.nonce = None
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def _send(self, status, payload):
                data = b"" if payload is None else json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):  # noqa: N802
                if self.path == "/jwks":
                    self._send(200, {"keys": [fake.key["jwk"]]})
                elif self.path == "/discovery":
                    self._send(200, {"revocation_endpoint": fake.base + "/revoke"})
                else:
                    self._send(404, {})

            def do_POST(self):  # noqa: N802
                form = dict(urllib.parse.parse_qsl(
                    self.rfile.read(int(self.headers["Content-Length"])).decode()))
                fake.posts.append((self.path, form))
                if self.path == "/revoke":
                    self._send(200, None)
                elif form.get("grant_type") == "refresh_token":
                    status, payload = fake.refresh_reply or (200, {
                        "access_token": ACCESS + "-2", "refresh_token": REFRESH + "-2",
                        "expires_in": 3600, "scope": fake.scope})
                    self._send(status, payload)
                else:
                    self._send(200, {
                        "access_token": ACCESS, "refresh_token": REFRESH, "expires_in": 3600,
                        "token_type": "Bearer", "scope": fake.scope,
                        "id_token": id_token(fake.key, nonce=fake.nonce, sub=fake.subject,
                                             aud=form["client_id"])})

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def auth(tmp_path, monkeypatch, key):
    monkeypatch.setattr(credentials, "credentials_dir", lambda: tmp_path / "credentials")
    fake = FakeAuth(key)
    monkeypatch.setattr(chatgpt_auth, "TOKEN_URL", fake.base + "/token")
    monkeypatch.setattr(chatgpt_auth, "JWKS_URL", fake.base + "/jwks")
    monkeypatch.setattr(chatgpt_auth, "DISCOVERY_URL", fake.base + "/discovery")
    yield fake
    fake.close()


def browser(fake, *, issue=ISSUED, error=None, state=None, seen=None):
    """Stand-in for the owner completing consent in their own browser."""
    def open_url(url):
        query = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query))
        if seen is not None:
            seen.append(query)
        fake.nonce = query["nonce"]
        reply = {"state": state or query["state"]}
        if error:
            reply["error"] = error
        else:
            reply.update(code="auth-code-1", scope=fake.scope)
            if issue:
                reply["client_id"] = issue
        with urllib.request.urlopen(query["redirect_uri"] + "?" + urllib.parse.urlencode(reply)):
            pass
    return open_url


def test_registration_then_returning_sign_in(auth, tmp_path):
    seen = []
    result = chatgpt_auth.sign_in(browser(auth, seen=seen), port=0)
    assert result == {"client_id": ISSUED, "plan_usage_granted": True,
                      "scopes": sorted(chatgpt_auth.SCOPES.split())}
    first = seen[0]
    assert first["client_id"] == "dynamic_agent_client"
    assert first["agent_name_hint"] == chatgpt_auth.APP_NAME
    assert first["ext_agent_host_id"].startswith("urn:uuid:")
    assert first["resource"] == "https://api.openai.com/v1" and first["response_type"] == "code"
    assert first["code_challenge_method"] == "S256"
    assert first["redirect_uri"].startswith("http://127.0.0.1:")
    assert first["redirect_uri"].endswith("/auth/callback")
    _, exchange = auth.posts[0]
    assert exchange["client_id"] == ISSUED and exchange["redirect_uri"] == first["redirect_uri"]
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(exchange["code_verifier"].encode()).digest()).rstrip(b"=").decode()
    assert challenge == first["code_challenge"]
    stored = (tmp_path / "credentials" / "openai-chatgpt.dpapi").read_bytes()
    assert ACCESS.encode() not in stored and REFRESH.encode() not in stored
    registration = chatgpt_auth.load_registration()
    assert registration["client_id"] == ISSUED and registration["plan_usage_granted"]
    assert chatgpt_auth.access_token() == ACCESS

    seen.clear()
    chatgpt_auth.sign_in(browser(auth, issue=None, seen=seen), port=0)
    again = seen[0]
    assert again["client_id"] == ISSUED and "agent_name_hint" not in again
    assert again["ext_agent_host_id"] == first["ext_agent_host_id"] and again["id_token_hint"]


def test_sign_in_refusals(auth):
    with pytest.raises(chatgpt_auth.SignInError, match="state"):
        chatgpt_auth.sign_in(browser(auth, state="forged"), port=0)
    with pytest.raises(chatgpt_auth.SignInError, match="access_denied"):
        chatgpt_auth.sign_in(browser(auth, error="access_denied"), port=0)
    with pytest.raises(chatgpt_auth.SignInError, match="incomplete"):
        chatgpt_auth.sign_in(browser(auth, issue=None), port=0)
    assert not credentials.exists(chatgpt_auth.PROVIDER_ID)
    chatgpt_auth.sign_in(browser(auth), port=0)
    auth.subject = "a-different-account"
    with pytest.raises(chatgpt_auth.SignInError, match="different ChatGPT account"):
        chatgpt_auth.sign_in(browser(auth, issue=None), port=0)
    assert chatgpt_auth.load_registration()["subject"] == "user-subject-1"


def test_plan_usage_not_granted_blocks_inference(auth):
    auth.scope = "openid profile email offline_access"
    result = chatgpt_auth.sign_in(browser(auth), port=0)
    assert result["plan_usage_granted"] is False
    with pytest.raises(credentials.CredentialError, match="not granted"):
        chatgpt_auth.access_token()


def _expire():
    record = json.loads(credentials.load(chatgpt_auth.PROVIDER_ID))
    record["expires_at"] = time.time() + 10
    credentials.store(chatgpt_auth.PROVIDER_ID, json.dumps(record, separators=(",", ":")))


def test_refresh_rotates_and_terminal_refresh_errors_clear_tokens(auth):
    chatgpt_auth.sign_in(browser(auth), port=0)
    _expire()
    assert chatgpt_auth.access_token() == ACCESS + "-2"
    path, form = auth.posts[-1]
    assert form == {"grant_type": "refresh_token", "client_id": ISSUED,
                    "refresh_token": REFRESH, "resource": "https://api.openai.com/v1"}
    assert chatgpt_auth.access_token() == ACCESS + "-2"  # no second refresh while fresh

    _expire()
    auth.refresh_reply = (503, {"error": "temporarily_unavailable"})
    with pytest.raises(OSError):
        chatgpt_auth.access_token()
    assert credentials.exists(chatgpt_auth.PROVIDER_ID)  # kept on a temporary failure

    auth.refresh_reply = (400, {"error": "invalid_grant"})
    with pytest.raises(credentials.CredentialError, match="sign in again"):
        chatgpt_auth.access_token()
    assert not credentials.exists(chatgpt_auth.PROVIDER_ID)
    assert chatgpt_auth.load_registration()["client_id"] == ISSUED


def test_sign_out_revokes_then_clears_tokens_and_keeps_registration(auth):
    chatgpt_auth.sign_in(browser(auth), port=0)
    host = chatgpt_auth.host_id()
    assert chatgpt_auth.sign_out() == {"tokens_cleared": True, "revoked": True}
    path, form = auth.posts[-1]
    assert path == "/revoke" and form == {"token": REFRESH, "token_type_hint": "refresh_token",
                                         "client_id": ISSUED}
    assert not credentials.exists(chatgpt_auth.PROVIDER_ID)
    assert chatgpt_auth.load_registration()["client_id"] == ISSUED
    assert chatgpt_auth.host_id() == host


# --- Responses translation --------------------------------------------------------------


def chat(messages):
    tools, _ = tool_definitions(META_TOOLS)
    return {"messages": messages, "tools": tools}


def sse(*events) -> bytes:
    return b"".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n".encode() for e in events)


def completed(output, **extra):
    return {"type": "response.completed", "response": {
        "model": "gpt-test-2026-11-01", "output": output,
        "usage": {"input_tokens": 300, "output_tokens": 20, "total_tokens": 320,
                  "output_tokens_details": {"reasoning_tokens": 12}}, **extra}}


REASONING_ITEM = {"type": "reasoning", "id": "rs_1",
                  "summary": [{"type": "summary_text", "text": REASONING}]}
CALL_ITEM = {"type": "function_call", "id": "fc_1", "call_id": "call_abc",
             "namespace": "laboratory", "name": "capabilities_search",
             "arguments": "{\"query\":\"math\"}", "status": "completed"}
TEXT_ITEM = {"type": "message", "role": "assistant", "content": [
    {"type": "output_text", "text": "The value is 42."}]}
CALL_STREAM = sse({"type": "response.created", "response": {}},
                  {"type": "response.output_text.delta", "delta": "x"},
                  completed([REASONING_ITEM, CALL_ITEM]))
TEXT_STREAM = sse(completed([TEXT_ITEM]))


def test_request_is_stateless_namespaced_and_omits_unsupported_fields():
    adapter = chatgpt.ChatGPTAdapter()
    messages = [
        {"role": "system", "content": "preamble"}, {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call_abc", "type": "function",
         "function": {"name": "capabilities_search", "arguments": "{\"query\":\"math\"}"}}]},
        {"role": "tool", "tool_call_id": "call_abc", "content": "{\"results\":[]}"},
    ]
    upstream = adapter.request(chat(messages), {
        "model": "gpt-test", "temperature": 0.7, "max_output_tokens": 2048, "seed": 41},
        {"inference": {"reasoning_effort": "low"}})
    body = upstream.body
    assert upstream.url == "https://api.openai.com/v1/responses"
    assert upstream.headers == {"Accept": "text/event-stream"}
    assert body["store"] is False and body["stream"] is True
    assert body["instructions"] == "preamble"
    for forbidden in ("temperature", "top_p", "max_output_tokens", "seed", "previous_response_id",
                      "metadata", "max_tool_calls", "user", "truncation"):
        assert forbidden not in body
    assert body["reasoning"] == {"effort": "low"}
    assert body["input"] == [
        {"role": "user", "content": "hi"},
        {"type": "function_call", "call_id": "call_abc", "namespace": "laboratory",
         "name": "capabilities_search", "arguments": "{\"query\":\"math\"}"},
        {"type": "function_call_output", "call_id": "call_abc", "output": "{\"results\":[]}"}]
    assert not any(item.get("role") == "system" for item in body["input"])
    (namespace,) = body["tools"]
    assert namespace["type"] == "namespace" and namespace["name"] == "laboratory"
    tools, _ = tool_definitions(META_TOOLS)
    assert [t["parameters"] for t in namespace["tools"]] == [
        t["function"]["parameters"] for t in tools]
    assert all(t["strict"] is False and t["type"] == "function" for t in namespace["tools"])
    assert adapter.auth_headers("tok") == {"Authorization": "Bearer tok"}
    with pytest.raises(ValueError):
        adapter.request(chat([]), {"model": "../x"}, {})


def test_stream_translation_hides_reasoning():
    adapter = chatgpt.ChatGPTAdapter()
    outcome = adapter.response(200, adapter.decode(200, CALL_STREAM, {}), {}, {})
    assert outcome.kind == "ok" and REASONING not in json.dumps(outcome.reply)
    (call,) = outcome.reply["choices"][0]["message"]["tool_calls"]
    assert call == {"id": "call_abc", "type": "function", "function": {
        "name": "capabilities_search", "arguments": "{\"query\":\"math\"}"}}
    assert outcome.usage == {"prompt_tokens": 300, "completion_tokens": 20,
                             "total_tokens": 320, "reasoning_tokens": 12}
    assert outcome.model_version == "gpt-test-2026-11-01"
    assert outcome.finish_reason == "tool_calls"
    text = adapter.response(200, adapter.decode(200, TEXT_STREAM, {}), {}, {})
    assert text.reply["choices"][0]["message"] == {"role": "assistant",
                                                   "content": "The value is 42."}
    incomplete = adapter.response(200, adapter.decode(200, sse({
        "type": "response.incomplete", "response": {
            "output": [TEXT_ITEM], "incomplete_details": {"reason": "content_filter"}}}), {}),
        {}, {})
    assert incomplete.kind == "ok" and incomplete.finish_reason == "incomplete:content_filter"


@pytest.mark.parametrize("stream,kind,code", [
    (sse({"type": "response.created", "response": {}}), "retry", "without_terminal"),
    (b"", "retry", "without_terminal"),
    (sse({"type": "response.failed", "response": {"error": {
        "code": "subscription_sharing_usage_limit_exceeded"}}}), "terminal", "usage_limit"),
    (sse({"type": "response.failed", "response": {"error": {
        "code": "subscription_sharing_usage_unavailable"}}}), "retry", "usage_unavailable"),
    (sse({"type": "error", "code": "server_error"}), "retry", "server_error"),
])
def test_stream_failures_are_classified(stream, kind, code):
    adapter = chatgpt.ChatGPTAdapter()
    outcome = adapter.response(200, adapter.decode(200, stream, {}), {}, {})
    assert outcome.kind == kind and code in outcome.code


@pytest.mark.parametrize("status,payload,kind,code", [
    (429, {"error": {"code": "subscription_sharing_usage_limit_exceeded"}}, "terminal",
     "see_chatgpt_settings_usage"),
    (403, {"error": {"code": "subscription_sharing_user_not_eligible"}}, "terminal",
     "not_eligible"),
    (400, {"error": {"code": "subscription_sharing_unsupported_capability",
                     "param": "temperature"}}, "terminal", "unsupported_capability"),
    (403, {"error": {"code": "subscription_sharing_route_not_supported"}}, "terminal", "route"),
    (401, {"error": {"code": "subscription_sharing_invalid_user"}}, "terminal", "invalid_user"),
    (403, {"error": {"code": "chatpass_v2_scope_not_authorized"}}, "terminal", "chatpass"),
    (503, {"error": {"code": "subscription_sharing_usage_unavailable"}}, "retry", "unavailable"),
    (503, {"error": {"code": "subscription_sharing_user_unavailable"}}, "retry", "user_unavail"),
    (429, {"error": {"code": "rate_limit_exceeded"}}, "retry", "rate_limit"),
    (503, {"detail": "direct routing unavailable"}, "retry", "admission"),
    (401, {"detail": "not accepted"}, "terminal", "admission"),
    (403, {"detail": "region"}, "terminal", "admission"),
    (500, None, "retry", "http_500"),
])
def test_http_errors_follow_the_documented_recovery(status, payload, kind, code):
    adapter = chatgpt.ChatGPTAdapter()
    raw = json.dumps(payload).encode() if payload is not None else b"<html>oops</html>"
    outcome = adapter.response(status, adapter.decode(status, raw, {}), {"Retry-After": "3"}, {})
    assert outcome.kind == kind and code in outcome.code
    if kind == "retry":
        assert outcome.retry_after_s == 3


def test_settings_record_every_difference():
    adapter = chatgpt.ChatGPTAdapter()
    support = adapter.settings_support({"temperature": 0.7, "seed": 41, "max_output_tokens": 9},
                                       {"reasoning_effort": "low"})
    assert support["temperature"].startswith("not sent")
    assert "no seed" in support["seed"] and support["max_output_tokens"].startswith("not sent")
    assert "reasoning.effort=low" in support["reasoning_effort"]
    assert "not sent" in adapter.settings_support({}, {})["reasoning_effort"]
    for bad in ({"reasoning_effort": "maximum"}, {"thinking_level": "high"}):
        with pytest.raises(ValueError):
            adapter.settings_support({}, bad)


# --- a full run through the broker -----------------------------------------------------


def test_full_run_against_a_fake_responses_server(tmp_path, monkeypatch, make_config):
    monkeypatch.setattr(credentials, "credentials_dir", lambda: tmp_path / "credentials")
    credentials.store(chatgpt_auth.PROVIDER_ID, json.dumps({
        "client_id": ISSUED, "access_token": ACCESS, "refresh_token": REFRESH,
        "id_token": None, "scopes": sorted(chatgpt_auth.SCOPES.split()),
        "expires_at": time.time() + 3600}, separators=(",", ":")))
    upstream = Upstream([(200, CALL_STREAM), (200, TEXT_STREAM)])
    monkeypatch.setattr(chatgpt, "ENDPOINT", upstream.url)
    adapter = chatgpt.ChatGPTAdapter()
    adapter.host = "127.0.0.1"
    monkeypatch.setitem(providers.ADAPTERS, chatgpt_auth.PROVIDER_ID, adapter)
    root = tmp_path / "data"
    shutil.copytree(FIXTURE_WORLD, root / "worlds/minimal/0.6.0")
    add_test_capabilities(root / "worlds/minimal/0.6.0")
    providers.write_profile(root, chatgpt_auth.PROVIDER_ID, enabled=True, per_run=5, per_day=10,
                            total=10)
    document = {
        "schema": "open-lab/model-provider", "schema_version": "1.0.0",
        "provider": {"id": "openai-chatgpt", "name": "OpenAI via ChatGPT plan usage",
                     "kind": "other", "locality": "external",
                     "api": {"protocol": "openai_chat_completions",
                             "base_url": "https://api.openai.com/v1"}},
        "models": [{"id": "orchestrator-chatgpt-test", "provider_ref": "gpt-test",
                    "roles": ["orchestrator", "text_generation"],
                    "modalities": {"input": ["text"], "output": ["text"]},
                    "context_window_tokens": 4096, "status": "available"}],
    }
    config = make_config(
        [], limits={"wall_clock_s": 60, "max_steps": 2},
        orchestrator={"provider_id": "openai-chatgpt", "model_id": "orchestrator-chatgpt-test",
                      "sampling": {"temperature": 0.7, "max_output_tokens": 2048, "seed": 41}},
        permissions={"filesystem": "workspace_write", "network": "loopback",
                     "models": ["orchestrator-chatgpt-test"]},
        extensions={"model_catalog": {"document": document,
                                      "sha256": sha256_text(canonical_json(document))}})
    try:
        result = run_experiment(config, root)
    finally:
        upstream.close()
    events = events_of(result.archive)
    results = [e["payload"] for e in events if e["type"] == "tool.result"]
    assert results and results[0]["status"] == "ok"
    assert len(upstream.requests) == 2
    assert all(r["headers"]["Authorization"] == f"Bearer {ACCESS}" for r in upstream.requests)
    second = upstream.requests[1]["body"]
    kinds = [item.get("type") for item in second["input"]]
    assert "function_call" in kinds and "function_call_output" in kinds
    assert "reasoning" not in kinds and "temperature" not in second
    for path in result.archive.rglob("*"):
        if path.is_file():
            text = path.read_bytes().decode("utf-8", "replace")
            for secret in (ACCESS, REFRESH, REASONING):
                assert secret not in text, path.name
    model = json.loads((result.archive / "model.json").read_text())
    assert model["settings"]["sampling_effective"]["temperature"].startswith("not sent")
    assert model["hosted"]["adapter"] == {"id": "openai-responses-chatgpt-plan",
                                          "version": "1.0.0", "host": "127.0.0.1"}
    reserved = [e for e in providers.Ledger(root, "openai-chatgpt").entries()
                if e["kind"] == "outcome"]
    assert [e["model_version"] for e in reserved] == ["gpt-test-2026-11-01"] * 2
    assert verify_archive(result.archive) == []
