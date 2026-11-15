"""OpenAI Responses API adapter for ChatGPT plan usage (S014d).

Translates the gateway's chat-completions conversation into a stateless
Responses request (``store: false``, ``stream: true``, full history in
``input``) and the streamed result back, without inventing, filling or dropping
tool arguments. The four meta-tools are sent unchanged as JSON Schema, with
``strict: false`` so the schema is not rewritten, grouped in one namespace as the
plan-usage preview requires.

The preview rejects ``temperature``, ``top_p`` and ``max_output_tokens``, and the
Responses API has no seed, so none of them is sent; each is recorded as a
provider-specific difference rather than emulated. Reasoning items are never
returned to the runtime, never resent and never recorded (the same rule as the
Gemini adapter's thought parts).

Translation version 1.0.0; any change to the mapping bumps it.
"""

from __future__ import annotations

import json
import re
from typing import Any

from . import chatgpt_auth, providers

ENDPOINT = "https://api.openai.com/v1/responses"
HOST = "api.openai.com"
NAMESPACE = "laboratory"
NAMESPACE_DESCRIPTION = "Tools of the Open Laboratory world."
_MODEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-]*$")
REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh")
NOT_SENT = {
    "temperature": "not sent: unsupported on the ChatGPT plan-usage route (preview)",
    "top_p": "not sent: unsupported on the ChatGPT plan-usage route (preview)",
    "max_output_tokens": "not sent: unsupported on the ChatGPT plan-usage route (preview)",
    "seed": "not sent: the Responses API has no seed parameter",
}
RETRY_STATUS = {408, 500, 502, 503, 504}
# Documented plan-usage codes that a later attempt may clear (errors-and-recovery).
RETRY_CODES = {"subscription_sharing_usage_unavailable", "subscription_sharing_user_unavailable",
               "server_error", "rate_limit_exceeded"}


def _event_code(response: Any) -> str:
    error = response.get("error") if isinstance(response, dict) else None
    return str(error.get("code") or "unknown") if isinstance(error, dict) else "unknown"


class ChatGPTAdapter:
    id = "openai-responses-chatgpt-plan"
    version = "1.0.0"
    host = HOST

    # --- credentials ---------------------------------------------------------------

    def secret(self) -> str:
        return chatgpt_auth.access_token()

    def auth_headers(self, secret: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {secret}"}

    # --- outgoing ------------------------------------------------------------------

    def request(self, chat: dict[str, Any], settings: dict[str, Any],
                state: dict[str, Any]) -> providers.Upstream:
        model = settings["model"]
        if not _MODEL.match(model):
            raise ValueError("invalid model slug")
        system, items = [], []
        for message in chat.get("messages", []):
            role, text = message.get("role"), message.get("content") or ""
            if role == "system":
                system.append(text)
            elif role == "user":
                items.append({"role": "user", "content": text})
            elif role == "assistant":
                if text:
                    items.append({"role": "assistant", "content": text})
                for call in message.get("tool_calls") or []:
                    items.append({"type": "function_call", "call_id": call.get("id"),
                                  "namespace": NAMESPACE, "name": call["function"]["name"],
                                  "arguments": call["function"].get("arguments") or "{}"})
            elif role == "tool":
                items.append({"type": "function_call_output",
                              "call_id": message.get("tool_call_id"), "output": text})
        body: dict[str, Any] = {"model": model, "input": items, "store": False, "stream": True}
        if system:
            body["instructions"] = "\n\n".join(system)
        functions = [
            {"type": "function", "name": tool["function"]["name"],
             "description": tool["function"]["description"],
             "parameters": tool["function"]["parameters"], "strict": False}
            for tool in chat.get("tools", []) if tool.get("type") == "function"
        ]
        if functions:
            body["tools"] = [{"type": "namespace", "name": NAMESPACE,
                              "description": NAMESPACE_DESCRIPTION, "tools": functions}]
        effort = state.get("inference", {}).get("reasoning_effort")
        if effort:
            body["reasoning"] = {"effort": effort}
        return providers.Upstream(ENDPOINT, body, {"Accept": "text/event-stream"})

    # --- incoming ------------------------------------------------------------------

    def decode(self, status: int, raw: bytes, headers: dict[str, str]) -> Any:
        """A list of stream events on 200; otherwise the JSON error body, if any."""
        text = raw.decode("utf-8", "replace")
        if status != 200:
            try:
                return json.loads(text) if text else None
            except ValueError:
                return None
        events = []
        for block in text.replace("\r\n", "\n").split("\n\n"):
            data = "\n".join(line[5:].lstrip() for line in block.split("\n")
                             if line.startswith("data:"))
            if not data or data == "[DONE]":
                continue
            try:
                events.append(json.loads(data))
            except ValueError:
                continue
        return events

    def response(self, status: int, payload: Any, headers: dict[str, str],
                 state: dict[str, Any]) -> providers.Outcome:
        if status != 200:
            return self._http_error(status, payload, headers)
        events = payload if isinstance(payload, list) else []
        terminal = next((e for e in reversed(events) if isinstance(e, dict) and e.get("type") in (
            "response.completed", "response.incomplete", "response.failed", "error")), None)
        if terminal is None:
            return providers.Outcome("retry", "stream_ended_without_terminal_event")
        kind = terminal["type"]
        if kind == "error":
            code = str(terminal.get("code") or "stream_error")
            return providers.Outcome("retry" if code in RETRY_CODES else "terminal",
                                     f"stream_error:{code}")
        response = terminal.get("response") or {}
        if kind == "response.failed":
            code = _event_code(response)
            return providers.Outcome("retry" if code in RETRY_CODES else "terminal",
                                     f"response_failed:{code}")
        return self._completed(response, state, kind)

    def _completed(self, response, state, kind) -> providers.Outcome:
        text, calls = [], []
        for item in response.get("output") or []:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "message":
                text.extend(part.get("text", "") for part in item.get("content") or []
                            if isinstance(part, dict) and part.get("type") == "output_text")
            elif item.get("type") == "function_call" and item.get("status", "completed") == \
                    "completed":
                calls.append({"id": item.get("call_id"), "type": "function", "function": {
                    "name": item.get("name", ""), "arguments": item.get("arguments") or "{}"}})
            # reasoning items are deliberately dropped here
        message: dict[str, Any] = {"role": "assistant", "content": "".join(text)}
        if calls:
            message["tool_calls"] = calls
        usage_in = response.get("usage") or {}
        usage = {ours: usage_in[theirs] for ours, theirs in (
            ("prompt_tokens", "input_tokens"), ("completion_tokens", "output_tokens"),
            ("total_tokens", "total_tokens")) if isinstance(usage_in.get(theirs), int)}
        reasoning = (usage_in.get("output_tokens_details") or {}).get("reasoning_tokens")
        if isinstance(reasoning, int):
            usage["reasoning_tokens"] = reasoning
        finish = "tool_calls" if calls else "stop"
        if kind == "response.incomplete":
            reason = (response.get("incomplete_details") or {}).get("reason", "unknown")
            finish = f"incomplete:{reason}"
        return providers.Outcome("ok", "ok", reply={"choices": [{"message": message}]},
                                 usage=usage, model_version=response.get("model"),
                                 finish_reason=finish)

    def _http_error(self, status, payload, headers) -> providers.Outcome:
        if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
            code = str(payload["error"].get("code") or payload["error"].get("type") or "unknown")
        else:
            # Direct-route admission may answer {"detail": "..."}: diagnostic text only.
            code = "admission"
        label = f"http_{status}:{code}"
        try:
            retry_after = float(headers.get("Retry-After") or headers.get("retry-after") or "")
        except ValueError:
            retry_after = None
        if code == "subscription_sharing_usage_limit_exceeded":
            return providers.Outcome("terminal", label + ":see_chatgpt_settings_usage")
        if code in RETRY_CODES or (status in RETRY_STATUS and code in ("admission", "unknown")):
            return providers.Outcome("retry", label, retry_after_s=retry_after)
        return providers.Outcome("terminal", label)

    # --- settings ------------------------------------------------------------------

    def settings_support(self, sampling: dict[str, Any],
                         inference: dict[str, Any]) -> dict[str, str]:
        unknown = set(inference) - {"reasoning_effort"}
        if unknown:
            raise ValueError(f"unsupported hosted_inference setting: {sorted(unknown)[0]}")
        effort = inference.get("reasoning_effort")
        if effort is not None and effort not in REASONING_EFFORTS:
            raise ValueError(f"reasoning_effort must be one of {', '.join(REASONING_EFFORTS)}")
        support = {name: NOT_SENT.get(name, "not sent: no Responses equivalent")
                   for name in sampling}
        support["reasoning_effort"] = (f"sent as reasoning.effort={effort}; effect not attested"
                                       if effort else "not sent; provider default")
        support["tool_schema"] = ("JSON Schema unchanged, strict=false, in namespace "
                                  f"'{NAMESPACE}' (preview requirement)")
        support["conversation_state"] = "store=false; full history resent each step"
        support["reasoning_channel"] = "reasoning items withheld, not resent and not recorded"
        return support


providers.ADAPTERS[chatgpt_auth.PROVIDER_ID] = ChatGPTAdapter()
