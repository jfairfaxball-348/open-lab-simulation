"""A single, supervisor-owned local specialist lease (ADR-0018).

The HTTP thread has no recorder. Its observations are drained by the supervisor.
No endpoint, model selector or permission is taken from a capability argument.
"""
from __future__ import annotations

import json
import queue
import threading
import time
import urllib.request

from jsonschema import Draft202012Validator

from . import ollama
from .model import NoRedirect, is_loopback
from .ownership import archive_lock
from .toolhost import permission_shortfall

INVOCATION = {"interface": "mcp", "server": "supervisor-gateway", "tool": "models.ask"}
BOUNDS = {"max_calls": (1, 32), "max_prompt_chars": (1, 6000),
          "max_output_chars": (1, 6000), "max_output_tokens": (1, 1024),
          "timeout_s": (1, 120), "lease_s": (1, 600)}


def policy(config, world):
    value = config.get("extensions", {}).get("specialist")
    if value is None:
        return None
    if (not isinstance(value, dict) or set(value) != {"model_id", *BOUNDS}
            or value["model_id"] != "specialist-text"):
        raise ValueError("specialist requires the fixed model id and all bounded fields")
    for key, (low, high) in BOUNDS.items():
        if type(value[key]) is not int or not low <= value[key] <= high:
            raise ValueError(f"invalid specialist {key}")
    if world.capabilities.get("models.ask", (None, {}))[1].get("invocation") != INVOCATION:
        raise ValueError("specialist requires the reserved models.ask capability")
    selected = [(d["provider"], m) for d in world.providers.values()
                for m in d["models"] if m["id"] == value["model_id"]]
    if len(selected) != 1:
        raise ValueError("specialist model must be uniquely bundled in the world")
    provider, model = selected[0]
    if (provider["kind"] != "ollama" or provider["locality"] != "local"
            or not is_loopback(provider["api"]["base_url"])
            or model["status"] != "available" or model["provider_ref"] != "llama3.2:3b"
            or not model.get("digest")):
        raise ValueError("specialist requires approved digest-pinned local llama3.2:3b")
    if config["orchestrator"]["provider_id"] != "scripted":
        # The selected orchestrator may come from a pinned external catalog.
        from .gateway import catalog
        documents, _ = catalog(config, world)
        document = documents.get(config["orchestrator"]["provider_id"], {})
        if document.get("provider", {}).get("locality") != "external":
            raise ValueError("specialist requires a hosted orchestrator in this milestone")
    return dict(value)


def admission(config, world, arguments):
    manifest = world.capabilities.get("models.ask", (None, {}))[1]
    caps = config.get("capabilities", {})
    if (manifest.get("invocation") != INVOCATION or "models.ask" in caps.get("deny", [])
            or ("allow" in caps and "models.ask" not in caps["allow"])):
        return "models.ask is not available"
    if permission_shortfall(manifest["permissions"], config["permissions"]):
        return "specialist permission not granted"
    if list(Draft202012Validator(manifest["input_schema"]).iter_errors(arguments)):
        return "invalid specialist input"
    return None


class Lease:
    def __init__(self, config, world, root, consumed=0):
        self.config, self.world, self.root = config, world, root
        self.policy = policy(config, world)
        self.consumed = consumed
        self.events = queue.Queue()
        self.jobs = queue.Queue()
        self.results = queue.Queue()
        self.busy = False
        self.thread = None
        self.stop = threading.Event()
        if self.policy:
            self.provider, self.model = next(
                (d["provider"], m) for d in world.providers.values()
                for m in d["models"] if m["id"] == self.policy["model_id"])
            self.base = ollama.native_base(self.provider["api"]["base_url"])

    def observation(self, stage, **data):
        self.events.put({"stage": stage, "model_id": self.policy["model_id"],
                         "digest": self.model["digest"], **data})

    def submit(self, arguments, reserve=lambda: None):
        problem = admission(self.config, self.world, arguments)
        if problem or not self.policy:
            return {"status": "denied", "error": problem or "specialist not configured"}
        if len(arguments["prompt"]) > self.policy["max_prompt_chars"]:
            return {"status": "error", "error": "specialist prompt exceeds configured limit"}
        if self.busy or self.consumed >= self.policy["max_calls"]:
            return {"status": "denied", "error": "specialist call budget exhausted or busy"}
        reserve()
        self.consumed += 1
        self.busy = True
        if self.thread is None:
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()
        self.jobs.put(dict(arguments))
        return None

    def poll(self):
        try:
            result = self.results.get_nowait()
        except queue.Empty:
            return None
        self.busy = False
        return result

    def _http(self, path, body=None, timeout=2):
        request = urllib.request.Request(
            self.base + path, data=None if body is None else json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        with opener.open(request, timeout=timeout) as response:
            # Bound wire data as well as the text exposed to the orchestrator.
            raw = response.read(131073)
        if len(raw) > 131072:
            raise ValueError("specialist response exceeds wire limit")
        return json.loads(raw)

    def _loop(self):
        held = None
        loaded = False
        touched = False
        expires = 0

        def release(reason):
            nonlocal held, loaded, touched
            if held is None:
                return
            start = time.monotonic()
            try:
                if touched:
                    self._http("/api/generate", {"model": self.model["provider_ref"],
                                               "keep_alive": 0})
                    self.observation("specialist_unloaded", reason=reason,
                                     duration_ms=int((time.monotonic()-start)*1000))
                loaded = False
            except Exception as error:
                self.observation("specialist_unload_failed", reason=type(error).__name__)
            finally:
                held.__exit__(None, None, None)
                held = None
                touched = False
                self.observation("specialist_lease_released", reason=reason)

        try:
            while not self.stop.is_set():
                if held is not None and time.monotonic() >= expires:
                    release("lease_timeout")
                try:
                    arguments = self.jobs.get(timeout=0.05)
                except queue.Empty:
                    continue
                start = time.monotonic()
                deadline = start + self.policy["timeout_s"]

                def remaining(deadline=deadline):
                    seconds = deadline - time.monotonic()
                    if seconds <= 0:
                        raise TimeoutError
                    return seconds

                try:
                    if held is None:
                        directory = self.root / "gpu-lease"
                        directory.mkdir(exist_ok=True)
                        candidate = archive_lock(directory, recover_stale=True)
                        candidate.__enter__()
                        held = candidate
                        self.observation("specialist_lease_acquired")
                        running = self._http("/api/ps", timeout=remaining())
                        if running.get("models"):
                            # Do not disturb another host user's resident model.
                            raise ValueError("Ollama already has a resident model")
                        identity = ollama.identify(self.base, self.model["provider_ref"],
                                                   timeout=remaining())
                        if identity.get("digest") != self.model["digest"]:
                            raise ValueError("specialist digest mismatch")
                    load_start = time.monotonic()
                    touched = True
                    self._http("/api/generate", {"model": self.model["provider_ref"],
                                               "keep_alive": self.policy["lease_s"]},
                               timeout=remaining())
                    if not loaded:
                        self.observation("specialist_loaded",
                                         duration_ms=int((time.monotonic()-load_start)*1000))
                    loaded = True
                    # Internal gateway chat request, translated to native Ollama to
                    # enforce finite residency on the inference request itself.
                    reply = self._http("/api/chat", {
                        "model": self.model["provider_ref"], "stream": False,
                        "messages": [{"role": "user", "content": arguments["prompt"]}],
                        "keep_alive": self.policy["lease_s"],
                        "options": {"num_predict": self.policy["max_output_tokens"],
                                    "num_ctx": 4096, "temperature": 0},
                    }, timeout=remaining())
                    text = reply["message"]["content"]
                    if not isinstance(text, str) or len(text) > self.policy["max_output_chars"]:
                        raise ValueError("specialist output exceeds configured limit")
                    expires = time.monotonic() + self.policy["lease_s"]
                    self.observation("specialist_reply", prompt=arguments["prompt"], text=text,
                                     duration_ms=int((time.monotonic()-start)*1000),
                                     usage={k: reply[k] for k in ("prompt_eval_count", "eval_count",
                                            "load_duration", "total_duration") if k in reply})
                    result = {"status": "ok", "result": {"text": text}}
                except Exception as error:
                    self.observation("specialist_failed", reason=type(error).__name__,
                                     duration_ms=int((time.monotonic()-start)*1000))
                    result = {"status": "timeout" if isinstance(error, TimeoutError) else "error",
                              "error": f"specialist request failed ({type(error).__name__})"}
                    release("call_failure")
                self.results.put(result)
        finally:
            release("segment_end")

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join(self.policy["timeout_s"] + 5)
            if self.thread.is_alive():
                self.observation("specialist_cleanup_pending", reason="finite server keep-alive")
