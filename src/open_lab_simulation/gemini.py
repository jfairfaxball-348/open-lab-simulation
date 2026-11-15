"""Google Gemini API adapter (``generateContent``), used for Gemma models (S014c).

Translates the gateway's chat-completions conversation to Gemini contents and
back, without inventing, filling or dropping tool arguments. The four meta-tool
schemas are sent unchanged as JSON Schema (``parametersJsonSchema``), so the
model sees what local models saw. Reasoning ("thought") parts are never returned
to the runtime, because the Laboratory never records a reasoning channel; thought
text is discarded and never sent back or written anywhere.

Thought signatures are not resent (translation 1.1.0). In S014c, resending them
after the runtime's context guard had dropped an earlier call turn made the API
return HTTP 500 for both Gemma 4 models, while the same history without
signatures was accepted; Gemma does not reject their absence. This is a
provider-specific condition recorded in each run's settings.

Translation version 1.1.0; any change to the mapping bumps it.
"""

from __future__ import annotations

import json
import re
from typing import Any

from . import providers
from .contracts import canonical_json, sha256_text

ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
HOST = "generativelanguage.googleapis.com"
_MODEL = re.compile(r"^[a-z0-9][a-z0-9.\-]*$")
THINKING_LEVELS = ("high", "minimal")
SAMPLING = {"temperature": "temperature", "top_p": "topP", "seed": "seed",
            "max_output_tokens": "maxOutputTokens"}
RETRYABLE = {408, 500, 502, 503, 504}


def _assistant_key(text: str, calls: list[dict[str, Any]]) -> str:
    ids = [call.get("id") for call in calls]
    return sha256_text(canonical_json(ids if any(ids) else {"text": text}))


def _duration(value: Any) -> float | None:
    if isinstance(value, str) and value.endswith("s"):
        try:
            return float(value[:-1])
        except ValueError:
            return None
    return None


class GeminiAdapter:
    id = "gemini-generate-content"
    version = "1.1.0"
    host = HOST

    # --- outgoing ------------------------------------------------------------------

    def request(self, chat: dict[str, Any], settings: dict[str, Any],
                state: dict[str, Any]) -> providers.Upstream:
        model = settings["model"]
        if not _MODEL.match(model):
            raise ValueError("invalid Gemini model id")
        system, contents = self._contents(chat.get("messages", []), state.get("originals", {}))
        body: dict[str, Any] = {"contents": contents}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        declarations = [
            {"name": tool["function"]["name"], "description": tool["function"]["description"],
             "parametersJsonSchema": tool["function"]["parameters"]}
            for tool in chat.get("tools", []) if tool.get("type") == "function"
        ]
        if declarations:
            body["tools"] = [{"functionDeclarations": declarations}]
        generation = {theirs: settings[ours] for ours, theirs in SAMPLING.items()
                      if ours in settings}
        level = state.get("inference", {}).get("thinking_level")
        if level:
            generation["thinkingConfig"] = {"thinkingLevel": level}
        if generation:
            body["generationConfig"] = generation
        return providers.Upstream(ENDPOINT.format(model=model), body)

    def _contents(self, messages, originals):
        system, contents, names = [], [], {}

        def add(role, parts):
            if contents and contents[-1]["role"] == role:
                contents[-1]["parts"].extend(parts)
            else:
                contents.append({"role": role, "parts": list(parts)})

        for message in messages:
            role, text = message.get("role"), message.get("content") or ""
            if role == "system":
                system.append(text)
            elif role == "user":
                add("user", [{"text": text}])
            elif role == "assistant":
                calls = message.get("tool_calls") or []
                for call in calls:
                    names[call.get("id")] = call["function"]["name"]
                original = originals.get(_assistant_key(text, calls))
                if original is not None:
                    add("model", original)
                    continue
                parts = [{"text": text}] if text else []
                for call in calls:
                    try:
                        args = json.loads(call["function"].get("arguments") or "{}")
                    except ValueError:
                        args = {}
                    part = {"name": call["function"]["name"],
                            "args": args if isinstance(args, dict) else {}}
                    if call.get("id"):
                        part["id"] = call["id"]
                    parts.append({"functionCall": part})
                add("model", parts or [{"text": ""}])
            elif role == "tool":
                try:
                    result = json.loads(text)
                except ValueError:
                    result = text
                response = result if isinstance(result, dict) else {"content": result}
                call_id = message.get("tool_call_id")
                part = {"name": names.get(call_id, "unknown"), "response": response}
                if call_id:
                    part["id"] = call_id
                add("user", [{"functionResponse": part}])
        return "\n\n".join(system), contents

    def auth_headers(self, secret: str) -> dict[str, str]:
        return {"x-goog-api-key": secret}

    # --- incoming ------------------------------------------------------------------

    def response(self, status: int, payload: Any, headers: dict[str, str],
                 state: dict[str, Any]) -> providers.Outcome:
        if status != 200:
            return self._error(status, payload, headers)
        if not isinstance(payload, dict):
            return providers.Outcome("retry", "invalid_response_body")
        usage_in = payload.get("usageMetadata") or {}
        usage = {ours: usage_in[theirs] for ours, theirs in (
            ("prompt_tokens", "promptTokenCount"), ("completion_tokens", "candidatesTokenCount"),
            ("reasoning_tokens", "thoughtsTokenCount"), ("total_tokens", "totalTokenCount"))
            if isinstance(usage_in.get(theirs), int)}
        candidates = payload.get("candidates") or []
        if not candidates:
            blocked = (payload.get("promptFeedback") or {}).get("blockReason", "no_candidate")
            return providers.Outcome(
                "ok", "ok", reply={"choices": [{"message": {"role": "assistant", "content": ""}}]},
                usage=usage, model_version=payload.get("modelVersion"),
                finish_reason=f"blocked:{blocked}")
        candidate = candidates[0]
        parts = (candidate.get("content") or {}).get("parts") or []
        text = "".join(p["text"] for p in parts
                       if isinstance(p.get("text"), str) and not p.get("thought"))
        calls, counter = [], state.get("calls", 0)
        for part in parts:
            call = part.get("functionCall")
            if not isinstance(call, dict):
                continue
            counter += 1
            ident = call.get("id") or f"gemini-call-{counter}"
            calls.append({"id": ident, "type": "function", "function": {
                "name": call.get("name", ""), "arguments": json.dumps(call.get("args") or {})}})
        message: dict[str, Any] = {"role": "assistant", "content": text}
        if calls:
            message["tool_calls"] = calls
        # Ids we generated must also appear in the resent original parts. Thought text
        # and thought signatures are dropped here and never travel back.
        original = [{k: v for k, v in p.items() if k != "thoughtSignature"}
                    for p in parts if not p.get("thought")]
        generated = iter(c["id"] for c in calls)
        for part in original:
            if isinstance(part.get("functionCall"), dict):
                part["functionCall"] = {**part["functionCall"], "id": next(generated)}
        originals = dict(state.get("originals", {}))
        originals[_assistant_key(text, calls)] = original
        return providers.Outcome(
            "ok", "ok", reply={"choices": [{"message": message}]}, usage=usage,
            model_version=payload.get("modelVersion"),
            finish_reason=candidate.get("finishReason"),
            state={"originals": originals, "calls": counter})

    def _error(self, status, payload, headers) -> providers.Outcome:
        error = payload.get("error", {}) if isinstance(payload, dict) else {}
        code = f"http_{status}:{error.get('status', 'unknown')}"
        details = error.get("details") or []
        retry_after = None
        daily = False
        for detail in details if isinstance(details, list) else []:
            if not isinstance(detail, dict):
                continue
            retry_after = _duration(detail.get("retryDelay")) or retry_after
            for violation in detail.get("violations") or []:
                if "PerDay" in str(violation.get("quotaId", "")):
                    daily = True
        if status == 429 and not daily:
            return providers.Outcome("retry", code, retry_after_s=retry_after)
        if status == 429:
            return providers.Outcome("terminal", code + ":daily_quota")
        if status in RETRYABLE:
            return providers.Outcome("retry", code, retry_after_s=retry_after)
        return providers.Outcome("terminal", code)

    def settings_support(self, sampling: dict[str, Any],
                         inference: dict[str, Any]) -> dict[str, str]:
        unknown = set(inference) - {"thinking_level"}
        if unknown:
            raise ValueError(f"unsupported hosted_inference setting: {sorted(unknown)[0]}")
        level = inference.get("thinking_level")
        if level is not None and level not in THINKING_LEVELS:
            raise ValueError("thinking_level must be 'high' or 'minimal'")
        support = {name: "sent as generationConfig." + SAMPLING[name] + "; effect not attested"
                   for name in sampling if name in SAMPLING}
        support["thinking_level"] = (f"sent as thinkingConfig.thinkingLevel={level}"
                                     if level else "not sent; provider default")
        support["tool_schema"] = "JSON Schema via parametersJsonSchema, unchanged"
        support["reasoning_channel"] = "thought parts withheld from the runtime and not recorded"
        support["thought_signatures"] = "not resent (provider returned HTTP 500 when resent)"
        return support


providers.ADAPTERS["google-gemini"] = GeminiAdapter()
