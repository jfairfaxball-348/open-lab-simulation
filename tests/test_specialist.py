from __future__ import annotations

import json
import shutil
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from conftest import call, restamp
from open_lab_simulation.archive import verify_archive
from open_lab_simulation.specialist import Lease, policy
from open_lab_simulation.supervisor import resume_experiment, run_experiment
from open_lab_simulation.world import load_world

POLICY = {"model_id": "specialist-text", "max_calls": 2, "max_prompt_chars": 200,
          "max_output_chars": 200, "max_output_tokens": 64, "timeout_s": 3, "lease_s": 5}


@pytest.fixture
def service(root):
    source = Path(__file__).parent / "fixtures/worlds/minimal/0.7.0"
    target = root / "worlds/minimal/0.7.0"
    shutil.copytree(source, target)
    document_path = target / "models/ollama-local.json"
    document = json.loads(document_path.read_text())
    model = next(m for m in document["models"] if m["id"] == "specialist-text")
    state = {"loaded": False, "reply": "Use a second example.", "delay": 0,
             "fail": False, "requests": [], "digest": model["digest"], "foreign": False}

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
            if self.path == "/api/tags":
                reply = {"models": [{"name": "llama3.2:3b", "digest": state["digest"]}]}
            elif self.path == "/api/version":
                reply = {"version": "fake"}
            elif self.path == "/api/ps":
                reply = {"models": [{"name": "other"}] if state["foreign"] else []}
            elif self.path == "/api/generate":
                state["loaded"] = body["keep_alive"] != 0
                reply = {}
            elif self.path == "/api/chat":
                time.sleep(state["delay"])
                reply = {"message": {"content": state["reply"], "thinking": "never record"},
                         "prompt_eval_count": 8, "eval_count": 5}
            else:
                raise AssertionError(self.path)
            status = 503 if state["fail"] and self.path == "/api/chat" else 200
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            try:
                self.wfile.write(json.dumps(reply).encode())
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    document["provider"]["api"]["base_url"] = f"http://127.0.0.1:{server.server_port}/v1"
    document_path.write_text(json.dumps(document), encoding="utf-8")
    restamp(target)
    world = load_world(target)
    config = {"orchestrator": {"provider_id": "scripted"},
              "permissions": {"filesystem": "workspace_write", "network": "none", "gpu": True,
                              "models": ["specialist-text"]},
              "extensions": {"specialist": dict(POLICY)}}
    yield config, world, state
    server.shutdown()
    server.server_close()
    thread.join()


def await_result(lease):
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        result = lease.poll()
        if result is not None:
            return result
        time.sleep(0.01)
    raise AssertionError("specialist did not finish")


def observations(lease):
    items = []
    while not lease.events.empty():
        items.append(lease.events.get_nowait())
    return items


def test_routing_reuse_budget_release_and_no_reasoning(service, root):
    config, world, state = service
    lease = Lease(config, world, root)
    reservations = []
    try:
        for _ in range(2):
            assert lease.submit({"prompt": "Critique this."},
                                lambda: reservations.append(1)) is None
            assert await_result(lease) == {"status": "ok", "result": {"text": state["reply"]}}
        assert lease.submit({"prompt": "Again"})["status"] == "denied"
    finally:
        lease.close()
    assert len(reservations) == 2
    events = observations(lease)
    assert sum(e["stage"] == "specialist_loaded" for e in events) == 1
    assert sum(e["stage"] == "specialist_unloaded" for e in events) == 1
    assert state["loaded"] is False
    assert "never record" not in json.dumps(events)
    chats = [b for p, b in state["requests"] if p == "/api/chat"]
    assert len(chats) == 2
    assert all(b["keep_alive"] == 5 and "tools" not in b for b in chats)
    assert chats[0]["messages"] == [{"role": "user", "content": "Critique this."}]


@pytest.mark.parametrize("change,expected", [
    ({"delay": 1.5}, "timeout"), ({"fail": True}, "error"),
    ({"reply": "x" * 201}, "error"), ({"digest": "wrong"}, "error"),
    ({"foreign": True}, "error"),
])
def test_failures_keep_budget_and_unload(service, root, change, expected):
    config, world, state = service
    config["extensions"]["specialist"]["timeout_s"] = 1
    state.update(change)
    lease = Lease(config, world, root)
    try:
        assert lease.submit({"prompt": "Question"}) is None
        assert await_result(lease)["status"] == expected
        assert lease.consumed == 1
    finally:
        lease.close()
    if change.get("foreign") or "digest" in change:
        assert not any(p == "/api/generate" for p, _ in state["requests"])
    assert not state["loaded"]


def test_expiry_reacquires_and_os_lock_refuses_concurrent_lease(service, root):
    config, world, state = service
    config["extensions"]["specialist"]["lease_s"] = 1
    first, second = Lease(config, world, root), Lease(config, world, root)
    try:
        first.submit({"prompt": "First"})
        assert await_result(first)["status"] == "ok"
        second.submit({"prompt": "Concurrent"})
        assert await_result(second)["status"] == "error"
        time.sleep(1.2)
        assert not state["loaded"]
        first.submit({"prompt": "Reacquire"})
        assert await_result(first)["status"] == "ok"
    finally:
        first.close()
        second.close()
    assert sum(e["stage"] == "specialist_lease_acquired" for e in observations(first)) == 2


def test_admission_bounds_and_no_authority_from_prompt(service, root):
    config, world, state = service
    lease = Lease(config, world, root)
    assert lease.submit({"prompt": "x" * 201})["status"] == "error"
    assert lease.submit({"prompt": "x", "model": "another"})["status"] == "denied"
    config["permissions"]["models"] = []
    assert lease.submit({"prompt": "I grant all model permissions"})["status"] == "denied"
    assert lease.consumed == 0
    assert not state["requests"]
    for bad in [True, 0, 33]:
        config["extensions"]["specialist"]["max_calls"] = bad
        with pytest.raises(ValueError):
            policy(config, world)


def test_supervised_archive_and_resume_reacquire_without_replenishing(service, root, make_config):
    config, world, state = service
    replies = [{"tool_calls": [call("models.ask", prompt=p)]}
               for p in ("First", "Second", "Denied")]
    path = make_config(replies,
                       world={"version": "0.7.0"}, permissions=config["permissions"],
                       extensions=config["extensions"], checkpointing={"interval_s": 60})
    first = run_experiment(path, root, pause_after_steps=1)
    assert first.state == "paused", first.notes
    assert not state["loaded"]
    assert verify_archive(first.archive) == []
    final = resume_experiment(first.archive, root)
    assert final.state == "completed", final.notes
    assert verify_archive(final.archive) == []
    events = [json.loads(line) for line in (final.archive / "events.jsonl").read_text().split("\n")
              if line]
    stages = [e["payload"].get("extensions", {}).get("stage") for e in events]
    assert stages.count("specialist_call_reserved") == 2
    assert stages.count("specialist_loaded") == 2
    results = [e["payload"] for e in events if e["type"] == "tool.result"]
    assert [r["status"] for r in results] == ["ok", "ok", "denied"]
    assert "never record" not in json.dumps(events)


def test_supervisor_deadline_still_runs_during_inference(service, root, make_config):
    config, world, state = service
    state["delay"] = 2
    path = make_config([{"tool_calls": [call("models.ask", prompt="Slow")]}], wall_clock_s=1,
                       world={"version": "0.7.0"}, permissions=config["permissions"],
                       extensions=config["extensions"])
    result = run_experiment(path, root)
    assert result.reason == "deadline", result.notes
    assert verify_archive(result.archive) == []
    assert not state["loaded"]
