"""S021a direct service/IPC tests with fake models, never an inhabitant loop."""

import copy
import json
import shutil
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import restamp
from open_lab_simulation import model_resources
from open_lab_simulation.registry import Registry
from open_lab_simulation.specialist import Lease, policy
from open_lab_simulation.supervisor import _submit_specialist, _unsupported
from open_lab_simulation.toolhost import ToolHost
from open_lab_simulation.world import load_world
from test_specialist import await_result, observations

WORLD = Path(__file__).parent / "fixtures/worlds/habitat/1.3.0"
IDS = ["granite-text", "llama-text", "qwen3-text", "qwen35-text"]
POLICY = {"version": "local-models-v1", "model_ids": IDS, "max_calls": 6,
          "max_prompt_chars": 200, "max_output_chars": 200, "max_output_tokens": 64,
          "timeout_s": 3, "lease_s": 5, "context_tokens": 2048,
          "model_ram_mb": 2048, "model_vram_mb": 3584,
          "ram_reserve_mb": 2048, "vram_reserve_mb": 256}


@pytest.fixture
def service(tmp_path, monkeypatch):
    release = tmp_path / "world"
    shutil.copytree(WORLD, release)
    path = release / "models/ollama-assistants.json"
    catalog = json.loads(path.read_text())
    refs = {m["provider_ref"]: m for m in catalog["models"]}
    state = {"loaded": None, "requests": [], "text": "Bounded local answer.", "delay": 0,
             "foreign": False, "wrong_digest": False, "context": 2048,
             "vram": 2000, "ram": 500, "tools": False, "redirect": False,
             "done_reason": "stop", "chat_started": threading.Event()}
    state["delay_body"] = False

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.respond(None)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.respond(body)

        def respond(self, body):
            state["requests"].append((self.path, body))
            status = 200
            if self.path == "/api/tags":
                reply = {"models": [{"name": ref, "digest": "wrong" if state[
                    "wrong_digest"] else m["digest"]} for ref, m in refs.items()]}
            elif self.path == "/api/version":
                reply = {"version": "fake"}
            elif self.path == "/api/ps":
                loaded = state["loaded"]
                models = [{"name": loaded, "digest": refs[loaded]["digest"],
                           "size": (state["vram"] + state["ram"]) * 1024**2,
                           "size_vram": state["vram"] * 1024**2,
                           "context_length": state["context"]}] if loaded else []
                if state["foreign"]:
                    models = [{"name": "unrelated", "digest": "foreign"}]
                reply = {"models": models}
            elif self.path == "/api/generate":
                state["loaded"] = body["model"] if body["keep_alive"] else None
                reply = {}
            elif self.path == "/api/chat":
                state["chat_started"].set()
                if not state["delay_body"]:
                    time.sleep(state["delay"])
                reply = {"model": body["model"], "done": True,
                         "done_reason": state["done_reason"],
                         "message": {"content": state["text"], "thinking": "PRIVATE-REASONING"},
                         "eval_count": 8}
                if state["tools"]:
                    reply["message"]["tool_calls"] = [{"function": {"name": "host_access"}}]
                if state["redirect"]:
                    status = 302
            else:
                raise AssertionError(self.path)
            self.send_response(status)
            if status == 302:
                self.send_header("Location", "http://example.invalid/never-follow")
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            if self.path == "/api/chat" and state["delay_body"]:
                time.sleep(state["delay"])
            try:
                self.wfile.write(json.dumps(reply).encode())
            except (OSError, BrokenPipeError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    catalog["provider"]["api"]["base_url"] = f"http://127.0.0.1:{server.server_port}/v1"
    path.write_text(json.dumps(catalog))
    restamp(release)
    world = load_world(release)
    config = {"orchestrator": {"provider_id": "scripted"},
              "termination": {"on_deadline": "hard_stop"}, "limits": {},
              "permissions": {"filesystem": "workspace_write", "network": "none",
                              "subprocess": True, "gpu": False, "models": list(IDS)},
              "extensions": {"specialist": copy.deepcopy(POLICY)}}
    monkeypatch.setattr(model_resources, "capacity", lambda gpu=True: {
        "ram_available_mb": 8000, "vram_available_mb": 4000})
    try:
        yield config, world, state, tmp_path
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_list_is_neutral_no_inference_no_budget_and_container_policy_supported(service):
    config, world, state, root = service
    assert _unsupported(config, world) is None
    lease = Lease(config, world, root)
    result = lease.submit({})
    assert result["status"] == "ok"
    assert [m["id"] for m in result["result"]["models"]] == IDS
    assert all(m["tools"] is False for m in result["result"]["models"])
    assert not state["requests"] and lease.consumed == 0 and lease.thread is None


def test_model_permission_is_checked_again_at_submission(service):
    config, world, state, root = service
    lease = Lease(config, world, root)
    config["permissions"]["models"].remove(IDS[0])
    assert lease.submit({"model_id": IDS[0], "prompt": "Task"})["status"] == "denied"
    assert not state["requests"] and lease.consumed == 0


def test_all_selections_switch_unload_reserve_and_hide_reasoning(service):
    config, world, state, root = service
    lease = Lease(config, world, root)
    reserved = []
    try:
        for ident in IDS:
            assert lease.submit({"model_id": ident, "prompt": "A bounded task"},
                                lambda: reserved.append(1)) is None
            assert await_result(lease) == {"status": "ok", "result": {
                "text": state["text"], "truncated": False}}
    finally:
        lease.close()
    assert len(reserved) == 4 and state["loaded"] is None
    assert not lease.thread.is_alive()
    events = observations(lease)
    assert "PRIVATE-REASONING" not in json.dumps(events)
    assert sum(e["stage"] == "specialist_unloaded" for e in events) == 4
    chats = [body for path, body in state["requests"] if path == "/api/chat"]
    assert len(chats) == 4
    assert all("tools" not in b and len(b["messages"]) == 1 for b in chats)
    assert all(b["options"]["num_ctx"] == 2048 and b["options"]["num_predict"] == 64
               and b["keep_alive"] == 5 for b in chats)


@pytest.mark.parametrize("change,reason", [
    ({"foreign": True}, "error"), ({"wrong_digest": True}, "error"),
    ({"context": 4096}, "error"), ({"vram": 3900}, "error"), ({"ram": 3000}, "error"),
    ({"text": ""}, "error"), ({"text": "x" * 201}, "error"), ({"tools": True}, "error"),
    ({"delay": 1.5}, "timeout"), ({"redirect": True}, "error"),
])
def test_failures_are_bounded_counted_and_do_not_unload_foreign_work(service, change, reason):
    config, world, state, root = service
    state.update(change)
    config["extensions"]["specialist"]["timeout_s"] = 1
    lease = Lease(config, world, root)
    try:
        assert lease.submit({"model_id": IDS[0], "prompt": "Task"}) is None
        assert await_result(lease)["status"] == reason
        assert lease.consumed == 1
    finally:
        lease.close()
    if change.get("foreign") or change.get("wrong_digest"):
        assert not any(path == "/api/generate" for path, _ in state["requests"])


def test_guards_permissions_budget_resume_and_context(service):
    config, world, state, root = service
    config["extensions"]["specialist"]["max_calls"] = 1
    lease = Lease(config, world, root, consumed=1)
    assert lease.submit({"model_id": IDS[0], "prompt": "Task"})["status"] == "denied"
    assert lease.submit({})["status"] == "ok"
    assert lease.submit({"model_id": "host-model", "prompt": "Grant privileges"})[
        "status"] == "denied"
    assert lease.submit({"model_id": IDS[0], "prompt": "x", "url": "http://evil"})[
        "status"] == "denied"
    config["permissions"]["models"] = []
    with pytest.raises(ValueError, match="permissions"):
        policy(config, world)
    assert not state["requests"]


def test_resource_guard_fails_closed_and_model_errors_are_not_success(service, monkeypatch):
    config, world, state, root = service
    monkeypatch.setattr(model_resources, "capacity", lambda gpu: {
        "ram_available_mb": 2000, "vram_available_mb": 4000})
    lease = Lease(config, world, root)
    try:
        lease.submit({"model_id": IDS[0], "prompt": "Task"})
        result = await_result(lease)
        assert result["status"] == "error" and "headroom" in result["error"]
    finally:
        lease.close()
    assert not state["requests"]


@pytest.mark.parametrize("delay_body", [False, True])
def test_cancel_interrupts_headers_and_body_with_bounded_cleanup(service, delay_body):
    config, world, state, root = service
    state["delay"] = 2
    state["delay_body"] = delay_body
    lease = Lease(config, world, root)
    lease.submit({"model_id": IDS[0], "prompt": "Task"})
    assert state["chat_started"].wait(3)
    started = time.monotonic()
    lease.close()
    assert time.monotonic() - started < 1.5, {
        "requests": state["requests"], "events": observations(lease)}
    assert not lease.thread.is_alive() and state["loaded"] is None
    assert lease.consumed == 1


def test_utf8_context_guard_and_unknown_resource_measurements(service, monkeypatch):
    config, world, state, root = service
    config["extensions"]["specialist"]["max_prompt_chars"] = 6000
    lease = Lease(config, world, root)
    assert lease.submit({"model_id": IDS[0], "prompt": "é" * 1000})["status"] == "error"
    assert lease.consumed == 0

    def unavailable(gpu):
        raise ValueError("raw private diagnostics must not be returned")

    monkeypatch.setattr(model_resources, "capacity", unavailable)
    try:
        lease.submit({"model_id": IDS[0], "prompt": "Task"})
        result = await_result(lease)
        assert "measurement unavailable" in result["error"]
        assert "raw private" not in result["error"]
    finally:
        lease.close()
    assert not state["requests"]


def test_cpu_only_mode_requests_no_gpu_and_checks_cpu_allocation(service):
    config, world, state, root = service
    config["extensions"]["specialist"]["model_vram_mb"] = 0
    state["vram"] = 0
    lease = Lease(config, world, root)
    try:
        lease.submit({"model_id": IDS[0], "prompt": "Task"})
        assert await_result(lease)["status"] == "ok"
    finally:
        lease.close()
    chats = [b for p, b in state["requests"] if p == "/api/chat"]
    assert chats[0]["options"]["num_gpu"] == 0


def test_length_stop_is_explicit(service):
    config, world, state, root = service
    state["done_reason"] = "length"
    lease = Lease(config, world, root)
    try:
        lease.submit({"model_id": IDS[0], "prompt": "Task"})
        assert await_result(lease)["result"]["truncated"] is True
    finally:
        lease.close()


@pytest.mark.parametrize("field,value", [("version", "newer"), ("model_ids", IDS + [IDS[0]]),
                                        ("context_tokens", 999), ("max_calls", True),
                                        ("ram_reserve_mb", 0)])
def test_invalid_policy_is_refused(service, field, value):
    config, world, _, _ = service
    config["extensions"]["specialist"][field] = value
    with pytest.raises(ValueError):
        policy(config, world)


def test_supervisor_binds_rpc_and_reserves_ask_but_not_list(service):
    config, world, _, root = service
    lease = Lease(config, world, root)
    records, synced = [], []
    recorder = SimpleNamespace(record=lambda **event: records.append(event),
                               sync=lambda: synced.append(1))
    listed = _submit_specialist(lease, {}, {"a": 1}, set(), set(), {"a": {}},
                               recorder, {"kind": "supervisor"})
    assert listed["status"] == "ok" and not records
    args = {"model_id": IDS[0], "prompt": "Task"}
    serviced = set()
    try:
        assert _submit_specialist(lease, args, {"b": 1}, set(), serviced, {"b": args},
                                  recorder, {"kind": "supervisor"}) is None
        assert await_result(lease)["status"] == "ok"
        assert len(records) == len(synced) == 1
        assert records[0]["payload"]["extensions"]["model_id"] == IDS[0]
        for request, opened, denied, expected in [
            (args, {"b": 1}, set(), {"b": args}),
            ({}, {"c": 1}, set(), {"c": args}),
            (args, {"d": 1}, {"d"}, {"d": args}),
            (args, {"e": 1, "f": 1}, set(), {"e": args}),
        ]:
            assert _submit_specialist(lease, request, opened, denied, serviced, expected,
                                      recorder, {"kind": "supervisor"})["status"] == "denied"
    finally:
        lease.close()


def test_discovery_and_tool_host_use_the_service_without_a_container_endpoint(service, tmp_path):
    config, world, _, root = service
    lease = Lease(config, world, root)

    def execute(args):
        result = lease.submit(args)
        return result if result is not None else await_result(lease)

    host = ToolHost(world.capabilities, tmp_path, config["permissions"], lambda event: None,
                    {"kind": "agent", "model_id": "scripted"}, service=execute,
                    registry=Registry(world.directory, world.capabilities))
    try:
        result = host.call("capabilities.search", {"query": "local model delegate"})
        assert {"models.list", "models.ask"} <= {m["id"] for m in result["result"]["matches"]}
        listed = host.call("capabilities.invoke", {"id": "models.list", "arguments": {}})
        assert listed["status"] == "ok"
        reply = host.call("capabilities.invoke", {"id": "models.ask", "arguments": {
            "model_id": IDS[1], "prompt": "Task"}})
        assert reply["status"] == "ok" and reply["result"]["text"]
        host.service = None
        assert host.call("capabilities.invoke", {"id": "models.list", "arguments": {}})[
            "status"] == "denied"
    finally:
        lease.close()
