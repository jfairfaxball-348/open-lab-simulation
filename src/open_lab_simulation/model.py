"""The model interface the agent loop talks to, and its two implementations.

``ScriptedModel`` replays fixed replies so the pipeline can be tested without
a model; its events carry the model id ``scripted``. ``OpenAIChatModel`` talks
to a local server over the OpenAI chat-completions API with tool calling,
which is what Ollama, llama.cpp's server and LM Studio expose.

Only what a model emits as its reply and its tool calls is passed on. If a
server returns a separate reasoning channel it is ignored, not recorded.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.parse import urlsplit

from .contracts import canonical_json, sha256_text

SCRIPTED_MODEL_ID = "scripted"
REQUEST_TIMEOUT_S = 600
# Rough but safe for English and JSON: real tokenisers average more characters per token.
CHARS_PER_TOKEN = 3
DEFAULT_CONTEXT_TOKENS = 4096
DEFAULT_OUTPUT_TOKENS = 512
TRUNCATED_RESULT = json.dumps({
    "truncated_by_runtime": True,
    "note": "This older tool result was removed to fit the model's context window.",
})


class ModelError(Exception):
    """The model server could not produce a reply."""


@dataclass
class ToolCall:
    id: str
    name: str            # the function name as the model wrote it
    capability_id: Any   # what it resolves to; may be invalid, the tool host decides
    arguments: Any       # an object, or None if the arguments could not be read
    raw_arguments: str   # the argument text as the model wrote it


@dataclass
class ModelReply:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    # Token counts, and anything the client did to the request or reply. Recorded.
    details: dict[str, Any] = field(default_factory=dict)


class ModelClient(Protocol):
    model_id: str

    def chat(self, messages: list[dict[str, Any]]) -> ModelReply | None:
        """The next reply, or None when the model has nothing further to say.

        May shorten ``messages`` in place to fit a context window.
        """


class ScriptedModel:
    """Replays ``replies`` in order, ignoring the conversation.

    A reply is ``{"text": str, "tool_calls": [...], "delay_s": number}``; only
    ``text`` is required. ``delay_s`` stands in for the time a model takes.
    """

    model_id = SCRIPTED_MODEL_ID

    def __init__(self, replies: list[dict[str, Any]], start: int = 0):
        self._replies = list(replies)
        self._next = start

    @property
    def position(self):
        return self._next

    def chat(self, messages: list[dict[str, Any]]) -> ModelReply | None:
        if self._next >= len(self._replies):
            return None
        reply = self._replies[self._next]
        self._next += 1
        time.sleep(float(reply.get("delay_s", 0)))
        calls = []
        for index, call in enumerate(reply.get("tool_calls", [])):
            call = call if isinstance(call, dict) else {}
            arguments = call.get("arguments")
            calls.append(ToolCall(
                f"scripted-{self._next}-{index}", str(call.get("capability_id")),
                call.get("capability_id"), arguments, json.dumps(arguments),
            ))
        return ModelReply(str(reply.get("text", "")), calls)


# --- OpenAI-compatible chat model ---------------------------------------------

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ModelError("model server redirects are refused")


def is_loopback(url: str) -> bool:
    try:
        parts = urlsplit(url)
        return (parts.scheme in {"http", "https"} and parts.hostname in
                ("127.0.0.1", "localhost", "::1") and not parts.username
                and not parts.password and not parts.query and not parts.fragment)
    except ValueError:
        return False


def function_name(capability_id: str) -> str:
    """Capability ids contain dots; function names in the chat API may not."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", capability_id)


def tool_definitions(manifests: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict]:
    """Present capabilities as chat-API tools. Returns the tools and name -> capability id."""
    tools, names = [], {}
    for manifest in sorted(manifests, key=lambda m: m["id"]):
        name = function_name(manifest["id"])
        names[name] = manifest["id"]
        tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": f"{manifest['summary']}\n\n{manifest['description']}",
                "parameters": manifest["input_schema"],
            },
        })
    return tools, names


def read_arguments(raw: Any) -> tuple[Any, bool]:
    """Turn a model's tool arguments into an object. Returns (arguments, was_repaired).

    Small models often wrap the JSON in a code fence or in prose, or send an
    empty string for "no arguments". Those are repaired. Anything else that
    is not a JSON object yields None, and the call is recorded as malformed.
    """
    if isinstance(raw, dict):
        return raw, False
    if not isinstance(raw, str):
        return None, False
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = None
        if raw.strip() == "":
            return {}, True
        start, end = raw.find("{"), raw.rfind("}")
        if 0 <= start < end:
            try:
                value = json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                return None, False
            return (value, True) if isinstance(value, dict) else (None, False)
        return None, False
    if isinstance(value, str):  # arguments encoded twice
        inner, _ = read_arguments(value)
        return inner, inner is not None
    return (value, False) if isinstance(value, dict) else (None, False)


def resolve_name(name: Any, names: dict[str, str]) -> Any:
    """Map what the model called a tool to a capability id, tolerating common variants."""
    if not isinstance(name, str):
        return None
    candidate = name.strip().removeprefix("functions.")
    if candidate in names:
        return names[candidate]
    if candidate in names.values():
        return candidate
    return names.get(function_name(candidate), candidate)


def repair_nested_arguments(capability_id: Any, arguments: Any) -> tuple[Any, bool]:
    """Decode a JSON-encoded capability input object; never infer ids or fields."""
    if capability_id != "capabilities.invoke" or not isinstance(arguments, dict):
        return arguments, False
    repaired = False
    # Some clients/models wrap the complete invocation arguments one extra
    # time. Unwrap only the exact envelope; never supply a missing id/input.
    if set(arguments) == {"arguments"}:
        inner = arguments["arguments"]
        if (isinstance(inner, dict) and set(inner) == {"id", "arguments"}
                and isinstance(inner["id"], str)):
            arguments, repaired = inner, True
    if isinstance(arguments.get("arguments"), str):
        decoded, _ = read_arguments(arguments["arguments"])
        if isinstance(decoded, dict):
            return {**arguments, "arguments": decoded}, True
    return arguments, repaired


def parse_reply(message: dict[str, Any], names: dict[str, str]) -> tuple[str, list[ToolCall], int]:
    """Read text and tool calls out of an assistant message. Returns (text, calls, repairs)."""
    text = message.get("content") or ""
    text = text if isinstance(text, str) else json.dumps(text)
    calls: list[ToolCall] = []
    repairs = 0
    listed = message.get("tool_calls")
    for index, entry in enumerate(listed if isinstance(listed, list) else []):
        function = entry.get("function") if isinstance(entry, dict) else None
        function = function if isinstance(function, dict) else {}
        raw = function.get("arguments")
        arguments, repaired = read_arguments(raw)
        name = function.get("name")
        capability_id = resolve_name(name, names)
        arguments, nested_repaired = repair_nested_arguments(capability_id, arguments)
        repairs += repaired or nested_repaired
        calls.append(ToolCall(
            str(entry.get("id") or f"call-{index}") if isinstance(entry, dict) else f"call-{index}",
            name if isinstance(name, str) else "",
            capability_id, arguments,
            raw if isinstance(raw, str) else json.dumps(raw),
        ))
    if not calls:
        # A tool call written out as text instead of through the tool-call channel.
        candidate, _ = read_arguments(text)
        if isinstance(candidate, dict) and isinstance(candidate.get("name"), str):
            arguments = candidate.get("parameters", candidate.get("arguments"))
            capability_id = resolve_name(candidate["name"], names)
            if isinstance(arguments, dict) and capability_id in names.values():
                raw_arguments = json.dumps(arguments)
                arguments, _ = repair_nested_arguments(capability_id, arguments)
                calls.append(ToolCall("repaired-0", candidate["name"], capability_id,
                                      arguments, raw_arguments))
                repairs += 1
    return text, calls, repairs


def fit_context(
    messages: list[dict[str, Any]], fixed_chars: int, budget_chars: int
) -> tuple[int, int]:
    """Shorten the conversation in place until the request fits.

    First the oldest tool results are blanked. If that is not enough, the
    oldest whole turns are dropped: an assistant message goes together with
    its tool results, so no result is left without its call. The initial
    prompt (the first message) and the latest message are never removed; if
    those two alone do not fit, the request is sent as it is.

    Returns (tool results blanked, messages dropped). Doing this explicitly,
    and recording it, is the alternative to the server silently cutting the
    start of the prompt.
    """
    def size() -> int:
        return fixed_chars + sum(len(json.dumps(m)) for m in messages)

    # Fresh tool results must reach the model at least once. In particular a
    # just-described capability must not become a truncation notice. Preserve
    # the entire latest assistant/tool group and discard older turns instead.
    recent = len(messages) - 1
    if messages and messages[-1].get("role") == "tool":
        while recent > 0 and messages[recent].get("role") == "tool":
            recent -= 1
    truncated = 0
    for message in messages[:recent]:
        if size() <= budget_chars:
            break
        content = message.get("content")
        # Blanking only helps when the result is longer than the notice replacing it.
        if (message.get("role") == "tool" and isinstance(content, str)
                and len(content) > len(TRUNCATED_RESULT)):
            message["content"] = TRUNCATED_RESULT
            truncated += 1
    dropped = 0
    # The generic system preamble and the initial user premise both survive.
    protected = 2 if messages and messages[0].get("role") == "system" else 1
    while size() > budget_chars and len(messages) > protected + 1:
        end = protected + 1
        while end < len(messages) and messages[end].get("role") == "tool":
            end += 1
        if end >= len(messages):
            break  # the oldest turn runs into the latest message, which must stay
        dropped += end - protected
        del messages[protected:end]
    return truncated, dropped


class OpenAIChatModel:
    """A model behind a local OpenAI-compatible ``/chat/completions`` endpoint."""

    def __init__(
        self,
        model_id: str,
        base_url: str,
        provider_ref: str,
        manifests: list[dict[str, Any]],
        sampling: dict[str, Any] | None = None,
        context_tokens: int | None = None,
        record_input=None,
    ):
        if not is_loopback(base_url):
            raise ModelError("only a model server on this machine may be used")
        self.model_id = model_id
        self.records_inputs = True
        self._record_input = record_input
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._provider_ref = provider_ref
        self._tools, self._names = tool_definitions(manifests)
        self._sampling = sampling or {}
        output_tokens = self._sampling.get("max_output_tokens", DEFAULT_OUTPUT_TOKENS)
        self._budget_chars = max(
            ((context_tokens or DEFAULT_CONTEXT_TOKENS) - output_tokens) * CHARS_PER_TOKEN, 1
        )

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            self._url, data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        # No proxy: the request must go to this machine and nowhere else.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        with opener.open(request, timeout=REQUEST_TIMEOUT_S) as response:
            return json.loads(response.read().decode("utf-8"))

    def chat(self, messages: list[dict[str, Any]]) -> ModelReply | None:
        details: dict[str, Any] = {}
        before = sha256_text(canonical_json(messages))
        truncated, dropped = fit_context(
            messages, len(json.dumps(self._tools)), self._budget_chars
        )
        if truncated:
            details["context_truncated_tool_results"] = truncated
        if dropped:
            details["context_dropped_messages"] = dropped
        body: dict[str, Any] = {
            "model": self._provider_ref, "messages": messages, "tools": self._tools,
            "stream": False,
            "max_tokens": self._sampling.get("max_output_tokens", DEFAULT_OUTPUT_TOKENS),
        }
        for ours, theirs in (("temperature", "temperature"), ("top_p", "top_p"),
                             ("seed", "seed"), ("max_output_tokens", "max_tokens")):
            if ours in self._sampling:
                body[theirs] = self._sampling[ours]

        context = {
            "policy": "oldest-tools-turns-v1", "before_sha256": before,
            "after_sha256": sha256_text(canonical_json(messages)),
            "budget_chars": self._budget_chars,
            "truncated_tool_results": truncated, "dropped_messages": dropped,
            "protected_overflow": (len(json.dumps(self._tools)) + sum(
                len(json.dumps(m)) for m in messages)) > self._budget_chars,
        }
        last_error = "no reply"
        for attempt in range(2):
            try:
                if self._record_input:
                    self._record_input({"request": body, "context": context,
                                        "attempt": attempt + 1, "backend": "local-chat"})
                reply = self._post(body)
                message = reply["choices"][0]["message"]
                if not isinstance(message, dict):
                    raise TypeError("message is not an object")
                break
            except urllib.error.HTTPError as error:
                last_error = f"HTTP {error.code}"
                if error.code < 500:
                    raise ModelError(f"model server refused the request ({last_error})") from None
            except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
                last_error = type(error).__name__
            if attempt == 0:
                details["retries"] = 1
                time.sleep(1)
        else:
            raise ModelError(f"model server did not give a usable reply ({last_error})")

        text, calls, repairs = parse_reply(message, self._names)
        if repairs:
            details["repaired_tool_calls"] = repairs
        usage = reply.get("usage") if isinstance(reply.get("usage"), dict) else {}
        for ours, theirs in (("input_tokens", "prompt_tokens"),
                             ("output_tokens", "completion_tokens")):
            if isinstance(usage.get(theirs), int) and usage[theirs] >= 0:
                details[ours] = usage[theirs]
        return ModelReply(text, calls, details)
