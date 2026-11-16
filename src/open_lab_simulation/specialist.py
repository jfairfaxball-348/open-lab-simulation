"""Supervisor-owned bounded local assistant leases (ADR-0018/0028).

The HTTP thread has no recorder. Its observations are drained by the supervisor.
Arguments select only frozen model IDs, never endpoints or permissions.
"""
from __future__ import annotations

import http.client
import io
import json
import queue
import threading
import time
import urllib.request
from urllib.parse import urlsplit

from jsonschema import Draft202012Validator

from . import model_resources, ollama
from .model import NoRedirect, is_loopback
from .ownership import archive_lock
from .toolhost import permission_shortfall

INVOCATION = {"interface": "mcp", "server": "supervisor-gateway", "tool": "models.ask"}
BOUNDS = {"max_calls": (1, 32), "max_prompt_chars": (1, 6000),
          "max_output_chars": (1, 6000), "max_output_tokens": (1, 1024),
          "timeout_s": (1, 120), "lease_s": (1, 600)}
LOCAL_BOUNDS = {**BOUNDS, "context_tokens": (1024, 4096),
                "model_ram_mb": (128, 16384), "model_vram_mb": (0, 16384),
                "ram_reserve_mb": (1024, 16384), "vram_reserve_mb": (128, 4096)}
VERSION = "local-models-v1"
LIST = {**INVOCATION, "tool": "models.list"}


class LocalServiceError(ValueError):
    """A fixed public failure reason, never raw server/host diagnostics."""


class _CancellableRead(io.RawIOBase):
    """Poll inside the raw reader so timeouts do not poison HTTP's buffered parser."""

    def __init__(self, sock, stop, deadline):
        self.sock, self.stop, self.deadline = sock.dup(), stop, deadline
        self.sock.settimeout(0.1)

    def readable(self):
        return True

    def readinto(self, buffer):
        while True:
            if self.stop is not None and self.stop.is_set():
                raise InterruptedError("local request cancelled")
            if time.monotonic() >= self.deadline:
                raise TimeoutError
            try:
                return self.sock.recv_into(buffer)
            except TimeoutError:
                continue

    def close(self):
        if not self.closed:
            self.sock.close()
        super().close()


class _CancellableSocket:
    def __init__(self, sock, stop, deadline):
        self.sock, self.stop, self.deadline = sock, stop, deadline

    def makefile(self, mode):
        if mode != "rb":
            raise ValueError("HTTP model responses are read-only")
        return io.BufferedReader(_CancellableRead(self.sock, self.stop, self.deadline))


def profiles(world, ids):
    selected = {}
    for ident in ids:
        matches = [(d["provider"], m) for d in world.providers.values()
                   for m in d["models"] if m["id"] == ident]
        if len(matches) != 1:
            raise ValueError("local model must have one bundled manifest")
        provider, model = matches[0]
        if (provider["kind"] != "ollama" or provider["locality"] != "local"
                or not is_loopback(provider["api"]["base_url"])
                or model["status"] != "available" or not model.get("digest")
                or "text" not in model["modalities"]["input"]
                or model["modalities"]["output"] != ["text"]):
            raise ValueError("local assistants require pinned local text profiles")
        selected[ident] = provider, model
    bases = {provider["api"]["base_url"] for provider, _ in selected.values()}
    if len(bases) != 1:
        raise ValueError("one local assistant provider is supported per lease")
    return selected


def local_policy(value, config, world):
    if set(value) != {"version", "model_ids", *LOCAL_BOUNDS} or value["version"] != VERSION:
        raise ValueError("invalid local model policy fields/version")
    ids = value["model_ids"]
    if (not isinstance(ids, list) or not 1 <= len(ids) <= 8
            or not all(isinstance(i, str) for i in ids) or len(set(ids)) != len(ids)):
        raise ValueError("local model ids must be a nonempty unique bounded list")
    for key, (low, high) in LOCAL_BOUNDS.items():
        if type(value[key]) is not int or not low <= value[key] <= high:
            raise ValueError(f"invalid local model {key}")
    if value["context_tokens"] <= value["max_output_tokens"] + 256:
        raise ValueError("context must leave room for input and framing")
    profiles(world, ids)
    if not set(ids) <= set(config["permissions"].get("models", [])):
        raise ValueError("local models require explicit model permissions")
    for ident, invocation in (("models.ask", INVOCATION), ("models.list", LIST)):
        manifest = world.capabilities.get(ident, (None, {}))[1]
        if manifest.get("invocation") != invocation or manifest["permissions"].get("gpu"):
            raise ValueError("local model capabilities require broker-only permissions")
    return dict(value)


def policy(config, world):
    value = config.get("extensions", {}).get("specialist")
    if value is None:
        return None
    if isinstance(value, dict) and "version" in value:
        result = local_policy(value, config, world)
        _orchestrator(config, world)
        return result
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
    _orchestrator(config, world)
    return dict(value)


def _orchestrator(config, world):
    if config["orchestrator"]["provider_id"] != "scripted":
        # The selected orchestrator may come from a pinned external catalog.
        from .gateway import catalog
        documents, _ = catalog(config, world)
        document = documents.get(config["orchestrator"]["provider_id"], {})
        if document.get("provider", {}).get("locality") != "external":
            raise ValueError("specialist requires a hosted orchestrator in this milestone")


def admission(config, world, arguments):
    listing = arguments == {} and config.get("extensions", {}).get("specialist", {}).get(
        "version") == VERSION
    ident = "models.list" if listing else "models.ask"
    invocation = LIST if listing else INVOCATION
    manifest = world.capabilities.get(ident, (None, {}))[1]
    caps = config.get("capabilities", {})
    if (manifest.get("invocation") != invocation or ident in caps.get("deny", [])
            or ("allow" in caps and ident not in caps["allow"])):
        return f"{ident} is not available"
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
        self.connection = None
        self.connection_lock = threading.Lock()
        self.multi = bool(self.policy and self.policy.get("version") == VERSION)
        if self.policy:
            self.models = profiles(world, self.policy["model_ids"]) if self.multi else {}
            self.provider, self.model = (next(iter(self.models.values())) if self.multi else next(
                (d["provider"], m) for d in world.providers.values()
                for m in d["models"] if m["id"] == self.policy["model_id"]))
            self.base = ollama.native_base(self.provider["api"]["base_url"])

    def observation(self, stage, **data):
        self.events.put({"stage": stage, "model_id": self.model["id"],
                         "digest": self.model["digest"], **data})

    def submit(self, arguments, reserve=lambda: None):
        problem = admission(self.config, self.world, arguments)
        if problem or not self.policy:
            return {"status": "denied", "error": problem or "specialist not configured"}
        if self.multi:
            if arguments == {}:
                return self.list_models()
            if (arguments.get("model_id") not in self.models
                    or arguments["model_id"] not in self.config["permissions"].get("models", [])):
                return {"status": "denied", "error": "local model id is not configured"}
            size = len(arguments["prompt"].encode("utf-8"))
            if size + self.policy["max_output_tokens"] + 256 > self.policy["context_tokens"]:
                return {"status": "error", "error": "prompt exceeds conservative context guard"}
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

    def list_models(self):
        return {"status": "ok", "result": {"models": [
            {"id": ident, "name": model["provider_ref"], "digest": model["digest"],
             "family": model.get("family", "unknown"),
             "quantization": model.get("quantization", "unknown"),
             "license": model.get("license", "See bundled notice"),
             "status": "configured", "tools": False}
            for ident, (_, model) in sorted(self.models.items())],
            "context_tokens": self.policy["context_tokens"],
            "max_output_tokens": self.policy["max_output_tokens"],
            "max_prompt_utf8_bytes": min(self.policy["max_prompt_chars"],
                self.policy["context_tokens"] - self.policy["max_output_tokens"] - 256)}}

    def poll(self):
        try:
            result = self.results.get_nowait()
        except queue.Empty:
            return None
        self.busy = False
        return result

    def _http(self, path, body=None, timeout=2, *, cleanup=False):
        if self.multi:
            parts = urlsplit(self.base)
            kind = (http.client.HTTPSConnection if parts.scheme == "https"
                    else http.client.HTTPConnection)
            connection = kind(parts.hostname, parts.port, timeout=min(timeout, 2))
            with self.connection_lock:
                self.connection = connection
            try:
                cleanup = cleanup or (
                    path == "/api/generate" and body and body.get("keep_alive") == 0)
                deadline = time.monotonic() + timeout
                stop = None if cleanup else self.stop
                connection.response_class = lambda sock, **kwargs: http.client.HTTPResponse(
                    _CancellableSocket(sock, stop, deadline), **kwargs)
                if self.stop.is_set() and not cleanup:
                    raise InterruptedError("local request cancelled")
                connection.request("GET" if body is None else "POST", parts.path + path,
                                   body=None if body is None else json.dumps(body).encode(),
                                   headers={"Content-Type": "application/json"})
                if self.stop.is_set() and not cleanup:
                    raise InterruptedError("local request cancelled")
                with connection.getresponse() as response:
                    if not 200 <= response.status < 300:
                        raise LocalServiceError(
                            "local HTTP response refused; no redirects followed")
                    raw = response.read(131073)
                if len(raw) > 131072:
                    raise LocalServiceError("specialist response exceeds wire limit")
                return json.loads(raw)
            finally:
                connection.close()
                with self.connection_lock:
                    if self.connection is connection:
                        self.connection = None
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
                    running = (self._http("/api/ps", cleanup=True).get("models", [])
                               if self.multi else [])
                    owned = not self.multi or any(
                        m.get("name") == self.model["provider_ref"]
                        and m.get("digest") == self.model["digest"] for m in running)
                    if owned:
                        self._http("/api/generate", {"model": self.model["provider_ref"],
                                                   "keep_alive": 0})
                        self.observation("specialist_unloaded", reason=reason,
                                         duration_ms=int((time.monotonic()-start)*1000))
                    else:
                        self.observation("specialist_unload_skipped", reason="identity not owned")
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
                    if self.multi:
                        chosen = arguments["model_id"]
                        if chosen != self.model["id"]:
                            release("model_switch")
                            self.provider, self.model = self.models[chosen]
                        self._headroom(before_load=not loaded)
                        if loaded:
                            self._allocation(remaining())
                    if self.stop.is_set():
                        raise InterruptedError("local model request cancelled")
                    if held is None:
                        directory = self.root / "gpu-lease"
                        directory.mkdir(exist_ok=True)
                        candidate = archive_lock(directory, recover_stale=True)
                        candidate.__enter__()
                        held = candidate
                        self.observation("specialist_lease_acquired")
                        if self.multi:
                            self._headroom(before_load=True)
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
                                               "keep_alive": self.policy["lease_s"],
                                               **({"options": self._options()}
                                                  if self.multi else {})},
                               timeout=remaining())
                    if not loaded:
                        self.observation("specialist_loaded",
                                         duration_ms=int((time.monotonic()-load_start)*1000))
                    loaded = True
                    if self.multi:
                        self._allocation(remaining())
                        self._headroom(before_load=False)
                    if self.stop.is_set():
                        raise InterruptedError("local model request cancelled")
                    # Internal gateway chat request, translated to native Ollama to
                    # enforce finite residency on the inference request itself.
                    reply = self._http("/api/chat", {
                        "model": self.model["provider_ref"], "stream": False,
                        "messages": [{"role": "user", "content": arguments["prompt"]}],
                        "keep_alive": self.policy["lease_s"],
                        "options": self._options(),
                    }, timeout=remaining())
                    if self.multi and (reply.get("done") is not True
                            or reply.get("model") != self.model["provider_ref"]):
                        raise LocalServiceError("incomplete or wrong-model response")
                    text = reply["message"]["content"]
                    if not isinstance(text, str) or len(text) > self.policy["max_output_chars"]:
                        raise ValueError("specialist output exceeds configured limit")
                    if self.multi and (not text.strip() or reply["message"].get("tool_calls")):
                        raise LocalServiceError(
                            "local model returned empty text or unexpected tools")
                    expires = time.monotonic() + self.policy["lease_s"]
                    self.observation("specialist_reply", prompt=arguments["prompt"], text=text,
                                     duration_ms=int((time.monotonic()-start)*1000),
                                     usage={k: reply[k] for k in ("prompt_eval_count", "eval_count",
                                            "load_duration", "total_duration") if k in reply},
                                     **({"done_reason": reply.get("done_reason")}
                                        if self.multi else {}))
                    result = {"status": "ok", "result": {"text": text,
                              **({"truncated": reply.get("done_reason") == "length"}
                                 if self.multi else {})}}
                except Exception as error:
                    self.observation("specialist_failed", reason=type(error).__name__,
                                     duration_ms=int((time.monotonic()-start)*1000))
                    detail = (str(error) if isinstance(error, LocalServiceError)
                              else type(error).__name__)
                    result = {"status": "timeout" if isinstance(error, TimeoutError) else "error",
                              "error": f"specialist request failed ({detail})"}
                    release("call_failure")
                self.results.put(result)
        finally:
            release("segment_end")

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join(5 if self.multi else self.policy["timeout_s"] + 5)
            if self.thread.is_alive():
                self.observation("specialist_cleanup_pending", reason="finite server keep-alive")

    def _options(self):
        return {"num_predict": self.policy["max_output_tokens"],
                "num_ctx": self.policy.get("context_tokens", 4096), "temperature": 0,
                **({"num_gpu": 0} if self.multi and self.policy["model_vram_mb"] == 0 else {})}

    def _headroom(self, *, before_load):
        policy = self.policy
        try:
            measured = model_resources.capacity(policy["model_vram_mb"] > 0)
        except Exception:
            raise LocalServiceError("resource measurement unavailable") from None
        self.observation("local_model_headroom", **measured, before_load=before_load)
        ram = policy["ram_reserve_mb"] + (policy["model_ram_mb"] if before_load else 0)
        vram = policy["vram_reserve_mb"] + (policy["model_vram_mb"] if before_load else 0)
        if (measured["ram_available_mb"] < ram or (policy["model_vram_mb"] > 0
                and measured["vram_available_mb"] < vram)):
            raise LocalServiceError("insufficient local model headroom")

    def _allocation(self, timeout):
        running = self._http("/api/ps", timeout=timeout).get("models")
        if not isinstance(running, list) or len(running) != 1:
            raise LocalServiceError("local residency changed or is not independently identifiable")
        active = running[0]
        if (active.get("name") != self.model["provider_ref"]
                or active.get("digest") != self.model["digest"]
                or active.get("context_length") != self.policy["context_tokens"]):
            raise LocalServiceError("local resident model identity/context mismatch")
        size, vram = active.get("size"), active.get("size_vram")
        if (type(size) is not int or type(vram) is not int or not 0 <= vram <= size):
            raise LocalServiceError("local allocation measurement unavailable")
        ram_mb, vram_mb = (size - vram) / 1024**2, vram / 1024**2
        self.observation("local_model_allocation", ram_mb=ram_mb, vram_mb=vram_mb,
                         context_tokens=active["context_length"])
        if (ram_mb > self.policy["model_ram_mb"] or vram_mb > self.policy["model_vram_mb"]):
            raise LocalServiceError("local model allocation exceeds configured budget")
