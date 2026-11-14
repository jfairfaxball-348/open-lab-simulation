"""Ollama's own API, used only for management: identity and measurement.

Chat goes through the OpenAI-compatible endpoint (``model.py``). This module
asks the server which exact model it holds, so a run archive can name it by
digest, and measures load time, memory and speed for the model-provider
manifest.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from .model import is_loopback


class OllamaError(Exception):
    pass


def native_base(openai_base_url: str) -> str:
    """The server root for the native API, given its OpenAI-compatible base URL."""
    return openai_base_url.rstrip("/").removesuffix("/v1")


def _request(base: str, path: str, body: dict[str, Any] | None = None, timeout: float = 30) -> Any:
    if not is_loopback(base):
        raise OllamaError("only a model server on this machine may be used")
    request = urllib.request.Request(
        base + path,
        data=None if body is None else json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="GET" if body is None else "POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError) as error:
        raise OllamaError(f"model server request failed ({type(error).__name__})") from None


def identify(base: str, provider_ref: str, *, timeout: float = 30) -> dict[str, Any]:
    """What the server holds under this name. Raises if the model is not present."""
    tags = _request(base, "/api/tags", timeout=timeout)
    wanted = {provider_ref, f"{provider_ref}:latest"}
    for entry in tags.get("models", []) if isinstance(tags, dict) else []:
        if isinstance(entry, dict) and wanted & {entry.get("name"), entry.get("model")}:
            identity: dict[str, Any] = {"provider_ref": provider_ref}
            for key in ("digest", "size", "modified_at"):
                if key in entry:
                    identity[key] = entry[key]
            details = entry.get("details")
            if isinstance(details, dict):
                identity["details"] = {
                    key: details[key]
                    for key in ("family", "parameter_size", "quantization_level", "format")
                    if key in details
                }
            try:
                identity["server_version"] = _request(base, "/api/version", timeout=timeout).get(
                    "version"
                )
            except (OllamaError, AttributeError):
                pass
            return identity
    raise OllamaError(f"model '{provider_ref}' is not present on the model server")


def measure(base: str, provider_ref: str) -> dict[str, Any]:
    """Unload the model, load it, generate once, and report what the server measured."""
    _request(base, "/api/generate", {"model": provider_ref, "keep_alive": 0}, timeout=120)
    prompt = "Write three short sentences about how rivers shape a landscape."
    done = _request(
        base, "/api/generate",
        {"model": provider_ref, "prompt": prompt, "stream": False,
         "options": {"num_predict": 128, "seed": 1, "temperature": 0}},
        timeout=900,
    )
    result: dict[str, Any] = {"provider_ref": provider_ref}
    nanoseconds = 1e9
    if done.get("load_duration"):
        result["load_s"] = round(done["load_duration"] / nanoseconds, 2)
    if done.get("eval_count") and done.get("eval_duration"):
        result["output_tokens"] = done["eval_count"]
        result["output_tokens_per_s"] = round(
            done["eval_count"] / (done["eval_duration"] / nanoseconds), 1
        )
    if done.get("prompt_eval_count") and done.get("prompt_eval_duration"):
        result["prompt_tokens_per_s"] = round(
            done["prompt_eval_count"] / (done["prompt_eval_duration"] / nanoseconds), 1
        )
    megabyte = 1024 * 1024
    running = _request(base, "/api/ps")
    for entry in running.get("models", []) if isinstance(running, dict) else []:
        if isinstance(entry, dict) and provider_ref in (entry.get("name"), entry.get("model")):
            if isinstance(entry.get("size"), int):
                result["loaded_mb"] = round(entry["size"] / megabyte)
            if isinstance(entry.get("size_vram"), int):
                result["vram_mb"] = round(entry["size_vram"] / megabyte)
                result["ram_mb"] = round((entry.get("size", 0) - entry["size_vram"]) / megabyte)
            if isinstance(entry.get("context_length"), int):
                result["context_length"] = entry["context_length"]
    return result
