"""A real model client, tested against a fake model server. No model is involved."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from conftest import call, events_of, restamp, tree
from open_lab_simulation import ollama
from open_lab_simulation.archive import verify_archive
from open_lab_simulation.cli import main
from open_lab_simulation.model import (
    TRUNCATED_RESULT,
    ModelError,
    OpenAIChatModel,
    fit_context,
    function_name,
    parse_reply,
    read_arguments,
    resolve_name,
    tool_definitions,
)
from open_lab_simulation.supervisor import LaunchError, run_experiment
from open_lab_simulation.world import file_sha256, load_world

NAMES = {"text_stats": "text.stats", "math_evaluate": "math.evaluate"}
DIGEST = "a80c4f17acd55265feec403c7aef86be0c25983ab279d83f3bcd3abbcb5b8b72"


class FakeServer:
    """Speaks just enough of the Ollama and OpenAI-compatible APIs for these tests."""

    def __init__(self):
        self.replies: list = []     # assistant messages, HTTP status codes, or ("delay", s)
        self.requests: list[dict] = []
        self.models = [{
            "name": "llama3.2:3b", "model": "llama3.2:3b", "digest": DIGEST, "size": 2019393189,
            "details": {"family": "llama", "parameter_size": "3.2B",
                        "quantization_level": "Q4_K_M", "host_path": "/secret/location"},
        }]
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *arguments):
                pass

            def _send(self, status: int, body: dict) -> None:
                data = json.dumps(body).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):  # noqa: N802
                if self.path == "/api/tags":
                    self._send(200, {"models": outer.models})
                elif self.path == "/api/version":
                    self._send(200, {"version": "0.0.0-fake"})
                elif self.path == "/api/ps":
                    self._send(200, {"models": [{
                        "name": "llama3.2:3b", "size": 3 * 1024**3, "size_vram": 2 * 1024**3,
                        "context_length": 4096}]})
                else:
                    self._send(404, {})

            def do_POST(self):  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/api/generate":
                    self._send(200, {} if "prompt" not in body else {
                        "load_duration": 2_500_000_000, "eval_count": 100,
                        "eval_duration": 4_000_000_000, "prompt_eval_count": 20,
                        "prompt_eval_duration": 100_000_000})
                    return
                outer.requests.append(body)
                reply = outer.replies.pop(0) if outer.replies else {"content": "Nothing more."}
                if isinstance(reply, tuple):
                    time.sleep(reply[1])
                    reply = {"content": "late"}
                if isinstance(reply, int):
                    self._send(reply, {"error": "fake failure"})
                    return
                self._send(200, {
                    "choices": [{"message": {"role": "assistant", **reply}}],
                    "usage": {"prompt_tokens": 100 + len(outer.requests), "completion_tokens": 7},
                })

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self._server.server_address[1]}"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def close(self):
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def server():
    fake = FakeServer()
    yield fake
    fake.close()


def point_world_at(release: Path, base_url: str, **model_changes) -> None:
    """Rewrite the world's provider manifest to name a server, keeping the release valid."""
    path = release / "models" / "ollama-local.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["provider"]["api"]["base_url"] = base_url
    document["models"][0].update(model_changes)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8", newline="\n")

    def update(manifest):
        manifest["model_manifests"][0]["sha256"] = file_sha256(path)

    restamp(release, update)


@pytest.fixture
def model_root(root, release, server) -> Path:
    point_world_at(release, server.base + "/v1")
    return root


@pytest.fixture
def model_config(make_config):
    def make(**changes) -> Path:
        base = {
            "orchestrator": {"provider_id": "ollama-local", "model_id": "orchestrator-small",
                             "sampling": {"temperature": 0, "seed": 7}},
            "permissions": {"network": "loopback", "models": ["orchestrator-small"]},
            "extensions": {},
        }
        base.update(changes)
        return make_config([], **base)
    return make


def tool_call(name: str, arguments, call_id: str = "c1") -> dict:
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": arguments}}


def model_calls(events: list[dict]) -> list[dict]:
    return [e["payload"] for e in events if e["type"] == "model.call"]


def tool_results(events: list[dict]) -> list[dict]:
    return [e["payload"] for e in events if e["type"] == "tool.result"]


# --- a run driven through the model client ------------------------------------

def test_run_with_a_model_server_uses_tools_and_is_recorded(model_root, model_config, server):
    server.replies = [
        {"content": "", "tool_calls": [
            tool_call("text_stats", json.dumps({"text": "hello world"}))]},
        {"content": "Now some arithmetic.", "tool_calls": [
            tool_call("math_evaluate", json.dumps({"expression": "1/2 + 1/4"}), "c2")]},
        {"content": "I have finished."},
    ]
    result = run_experiment(model_config(), model_root)
    assert (result.state, result.reason) == ("completed", "the model made no tool call")
    assert verify_archive(result.archive) == []

    events = events_of(result.archive)
    assert [r["status"] for r in tool_results(events)] == ["ok", "ok"]
    assert tool_results(events)[1]["result"] == {"value": 0.75, "exact": "3/4"}
    calls = model_calls(events)
    assert len(calls) == 3
    assert calls[0]["model_id"] == "orchestrator-small"
    assert (calls[0]["input_tokens"], calls[0]["output_tokens"]) == (101, 7)
    agent_events = [e for e in events if e.get("actor", {}).get("kind") == "agent"]
    assert agent_events and all(
        e["actor"] == {"kind": "agent", "model_id": "orchestrator-small",
                       "provider_id": "ollama-local"} for e in agent_events
    )
    assert [e["payload"]["text"] for e in events if e["type"] == "agent.output"] == [
        "", "Now some arithmetic.", "I have finished."
    ]


def test_requests_carry_only_the_prompt_the_tools_and_the_conversation(
    model_root, model_config, server
):
    server.replies = [{"content": "", "tool_calls": [
        tool_call("text_stats", json.dumps({"text": "a b"}))]}]
    run_experiment(model_config(), model_root)
    first, second = server.requests
    assert first["model"] == "llama3.2:3b" and first["stream"] is False
    assert (first["temperature"], first["seed"]) == (0, 7)
    # One prompt in: no system message, nothing added by the runtime.
    assert first["messages"] == [{"role": "user", "content": "You may use whatever you find."}]
    offered = {tool["function"]["name"] for tool in first["tools"]}
    assert {"text_stats", "math_evaluate"} <= offered
    assert all("." not in name for name in offered)
    assistant, result = second["messages"][1:]
    assert assistant["tool_calls"][0]["function"]["name"] == "text_stats"
    assert (result["role"], result["tool_call_id"]) == ("tool", "c1")
    assert json.loads(result["content"]) == {
        "status": "ok", "result": {"characters": 3, "words": 2, "lines": 1}
    }


def test_exact_model_identity_is_stored_in_the_archive(model_root, model_config):
    result = run_experiment(model_config(), model_root)
    stored = json.loads((result.archive / "model.json").read_text(encoding="utf-8"))
    assert stored["real_model"] is True
    assert stored["provider_ref"] == "llama3.2:3b"
    assert stored["identity"]["digest"] == DIGEST
    assert stored["identity"]["server_version"] == "0.0.0-fake"
    assert stored["sampling"] == {"temperature": 0, "seed": 7}
    assert "secret" not in json.dumps(stored), "only named fields are copied from the server"
    summary = json.loads((result.archive / "summary.json").read_text(encoding="utf-8"))
    assert summary["model"] == {
        "model_id": "orchestrator-small", "real_model": True, "provider_id": "ollama-local",
        "provider_ref": "llama3.2:3b", "digest": DIGEST,
    }


def test_malformed_tool_calls_are_repaired_or_recorded(model_root, model_config, server):
    server.replies = [
        {"content": "", "tool_calls": [
            tool_call("text_stats", '```json\n{"text": "fenced"}\n```', "a"),
            tool_call("text.stats", {"text": "already an object"}, "b"),
            tool_call("text_stats", "this is not json", "c"),
            tool_call("make_coffee", "{}", "d"),
        ]},
        {"content": '{"name": "math_evaluate", "parameters": {"expression": "2 ** 10"}}'},
        {"content": "done"},
    ]
    result = run_experiment(model_config(), model_root)
    events = events_of(result.archive)
    statuses = [(r["status"], r.get("error", "")) for r in tool_results(events)]
    assert statuses[0] == ("ok", "") and statuses[1] == ("ok", "")
    assert statuses[2] == ("error", "unknown capability 'make_coffee'")
    assert statuses[3] == ("ok", "")
    assert tool_results(events)[3]["result"] == {"value": 1024, "exact": "1024"}
    (malformed,) = [e for e in events if e["type"] == "error"]
    assert "malformed tool call" in malformed["payload"]["message"]
    # What the model wrote is kept verbatim; that a repair happened is on the record.
    assert model_calls(events)[0]["repaired_tool_calls"] == 1
    assert model_calls(events)[1]["repaired_tool_calls"] == 1
    assert [e["payload"]["text"] for e in events if e["type"] == "agent.output"][1].startswith(
        '{"name": "math_evaluate"'
    )
    assert verify_archive(result.archive) == []


def test_denied_capabilities_are_not_offered_to_the_model(model_root, model_config, server):
    run_experiment(model_config(capabilities={"deny": ["math.evaluate"]}), model_root)
    offered = {tool["function"]["name"] for tool in server.requests[0]["tools"]}
    assert "math_evaluate" not in offered and "text_stats" in offered


def test_reasoning_channel_is_never_recorded(model_root, model_config, server):
    server.replies = [{"content": "Visible reply.", "reasoning": "PRIVATE-CHAIN-OF-THOUGHT",
                       "reasoning_content": "PRIVATE-CHAIN-OF-THOUGHT"}]
    result = run_experiment(model_config(), model_root)
    recorded = "".join(p.read_text(encoding="utf-8", errors="ignore")
                       for p in result.archive.rglob("*") if p.is_file())
    assert "Visible reply." in recorded and "PRIVATE-CHAIN-OF-THOUGHT" not in recorded


def test_old_tool_results_are_dropped_to_fit_the_context(root, release, server, model_config):
    point_world_at(release, server.base + "/v1", context_window_tokens=1400)
    long_text = "word " * 400
    server.replies = [
        {"content": long_text, "tool_calls": [
            tool_call("text_stats", json.dumps({"text": long_text}), f"c{n}")]}
        for n in range(3)
    ]
    result = run_experiment(model_config(), root)
    calls = model_calls(events_of(result.archive))
    assert any(c.get("context_truncated_tool_results") for c in calls)
    dropped = [m for m in server.requests[-1]["messages"] if m.get("content") == TRUNCATED_RESULT]
    assert dropped and server.requests[-1]["messages"][0]["role"] == "user"
    assert verify_archive(result.archive) == []


def test_one_server_failure_is_retried_and_recorded(model_root, model_config, server):
    server.replies = [500, {"content": "recovered"}]
    result = run_experiment(model_config(), model_root)
    assert result.state == "completed"
    assert model_calls(events_of(result.archive))[0]["retries"] == 1


def test_persistent_server_failure_fails_the_run(model_root, model_config, server):
    server.replies = [500, 500]
    result = run_experiment(model_config(), model_root)
    assert result.state == "failed"
    events = events_of(result.archive)
    (error,) = [e for e in events if e["type"] == "error"]
    assert error["payload"]["origin"] == "model" and "HTTP 500" in error["payload"]["message"]
    assert events[-1]["payload"]["to"] == "failed"
    assert verify_archive(result.archive) == []


def test_deadline_applies_while_the_model_is_thinking(model_root, model_config, server):
    server.replies = [("delay", 30)]
    started = time.monotonic()
    result = run_experiment(model_config(limits={"wall_clock_s": 1}), model_root)
    assert time.monotonic() - started < 20
    assert (result.state, result.reason) == ("terminated", "deadline")


# --- refusing to launch -----------------------------------------------------------

def refused(config: Path, root: Path) -> str:
    before = tree(root)
    with pytest.raises(LaunchError) as raised:
        run_experiment(config, root)
    assert tree(root) == before
    return str(raised.value)


def test_model_must_be_granted_by_the_experiment(model_root, model_config, server):
    for permissions in ({"network": "none", "models": ["orchestrator-small"]},
                        {"network": "loopback", "models": []},
                        {"network": "loopback"}):
        config = model_config(permissions={"filesystem": "none", **permissions})
        assert "must grant" in refused(config, model_root)
    assert server.requests == []


def test_model_missing_from_the_server_is_refused(model_root, model_config, server):
    server.models = []
    assert "is not present" in refused(model_config(), model_root)


def test_unreachable_or_remote_model_server_is_refused(root, release, model_config):
    point_world_at(release, "http://127.0.0.1:9/v1")
    assert "request failed" in refused(model_config(), root)
    point_world_at(release, "http://models.example.org/v1")
    assert "must be on this machine" in refused(model_config(), root)


def test_unknown_model_is_refused(model_root, model_config):
    config = model_config(
        orchestrator={"model_id": "gpt-enormous"},
        permissions={"network": "loopback", "models": ["gpt-enormous"]},
    )
    assert "does not define exactly one model" in refused(config, model_root)


# --- parsing and repair, in isolation ----------------------------------------------

@pytest.mark.parametrize(
    "raw, expected, repaired",
    [
        ('{"text": "a"}', {"text": "a"}, False),
        ({"text": "a"}, {"text": "a"}, False),
        ("", {}, True),
        ("   ", {}, True),
        ('```json\n{"text": "a"}\n```', {"text": "a"}, True),
        ('Here you go: {"text": "a"} hope that helps', {"text": "a"}, True),
        ('"{\\"text\\": \\"a\\"}"', {"text": "a"}, True),
        ("not json", None, False),
        ("[1, 2]", None, False),
        ("{broken", None, False),
        (None, None, False),
        (5, None, False),
    ],
)
def test_reading_tool_arguments(raw, expected, repaired):
    assert read_arguments(raw) == (expected, repaired)


@pytest.mark.parametrize(
    "written, resolved",
    [("text_stats", "text.stats"), ("text.stats", "text.stats"),
     ("functions.math_evaluate", "math.evaluate"), (" text_stats ", "text.stats"),
     ("make_coffee", "make_coffee"), (None, None), (7, None)],
)
def test_resolving_tool_names(written, resolved):
    assert resolve_name(written, NAMES) == resolved


def test_text_is_only_treated_as_a_tool_call_when_it_names_a_real_tool():
    text, calls, repairs = parse_reply(
        {"content": '{"name": "make_coffee", "parameters": {}}'}, NAMES)
    assert (calls, repairs) == ([], 0) and text.startswith("{")
    _, calls, _ = parse_reply({"content": "I could call text_stats but will not."}, NAMES)
    assert calls == []
    _, calls, _ = parse_reply({"content": None, "tool_calls": "nonsense"}, NAMES)
    assert calls == []


def test_tool_definitions_come_from_the_manifests(release):
    world = load_world(release)
    manifests = [m for _, m in world.capabilities.values()]
    tools, names = tool_definitions(manifests)
    assert names["text_stats"] == "text.stats" and function_name("a.b-c") == "a_b-c"
    stats = next(t["function"] for t in tools if t["function"]["name"] == "text_stats")
    assert stats["parameters"] == world.capabilities["text.stats"][1]["input_schema"]
    assert world.capabilities["text.stats"][1]["summary"] in stats["description"]


def test_fit_context_blanks_oldest_tool_results_first():
    messages = [{"role": "user", "content": "prompt"}]
    for n in range(4):
        messages += [{"role": "assistant", "content": "x"},
                     {"role": "tool", "tool_call_id": str(n), "content": "r" * 1000}]
    assert fit_context(messages, 0, 10**6) == 0
    assert fit_context(messages, 0, 3000) == 2
    contents = [m["content"] for m in messages if m["role"] == "tool"]
    assert contents[:2] == [TRUNCATED_RESULT] * 2 and contents[2] == "r" * 1000
    assert messages[0]["content"] == "prompt"
    assert fit_context(messages, 0, 3000) == 0, "already-blanked results are not counted again"


def test_model_client_only_talks_to_this_machine():
    with pytest.raises(ModelError):
        OpenAIChatModel("m", "http://models.example.org/v1", "ref", [])
    with pytest.raises(ollama.OllamaError):
        ollama.identify("http://models.example.org", "ref")


# --- measurement -------------------------------------------------------------------

def test_measurement_reports_what_the_server_measured(server, capsys):
    assert ollama.native_base(server.base + "/v1/") == server.base
    measured = ollama.measure(server.base, "llama3.2:3b")
    assert measured == {
        "provider_ref": "llama3.2:3b", "load_s": 2.5, "output_tokens": 100,
        "output_tokens_per_s": 25.0, "prompt_tokens_per_s": 200.0,
        "loaded_mb": 3072, "vram_mb": 2048, "ram_mb": 1024, "context_length": 4096,
    }
    assert main(["measure-model", server.base + "/v1", "llama3.2:3b"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["identity"]["digest"] == DIGEST and printed["measured"]["vram_mb"] == 2048
    assert main(["measure-model", server.base + "/v1", "absent:1b"]) == 1


def test_scripted_runs_never_contact_a_model_server(model_root, make_config, server):
    result = run_experiment(make_config([{"text": "hi", "tool_calls": [
        call("text.stats", text="a")]}]), model_root)
    assert result.state == "completed" and server.requests == []
    stored = json.loads((result.archive / "model.json").read_text(encoding="utf-8"))
    assert stored == {"kind": "scripted", "model_id": "scripted", "real_model": False}
