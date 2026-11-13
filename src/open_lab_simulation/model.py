"""The model interface the agent loop talks to, and a scripted stand-in.

No real model is called anywhere in this package yet (that is S003b). The
scripted model replays a fixed list of replies so the whole pipeline can be
exercised and tested without one. Its events carry the model id ``scripted``,
so a scripted run can never be mistaken for a model's behaviour.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

SCRIPTED_MODEL_ID = "scripted"


@dataclass
class ModelReply:
    text: str
    # Each: {"capability_id": ..., "arguments": {...}}
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


class ModelClient(Protocol):
    model_id: str

    def chat(self, messages: list[dict[str, Any]]) -> ModelReply | None:
        """The next reply, or None when the model has nothing further to say."""


class ScriptedModel:
    """Replays ``replies`` in order, ignoring the conversation.

    A reply is ``{"text": str, "tool_calls": [...], "delay_s": number}``; only
    ``text`` is required. ``delay_s`` stands in for the time a model takes.
    """

    model_id = SCRIPTED_MODEL_ID

    def __init__(self, replies: list[dict[str, Any]]):
        self._replies = list(replies)
        self._next = 0

    def chat(self, messages: list[dict[str, Any]]) -> ModelReply | None:
        if self._next >= len(self._replies):
            return None
        reply = self._replies[self._next]
        self._next += 1
        time.sleep(float(reply.get("delay_s", 0)))
        return ModelReply(str(reply.get("text", "")), list(reply.get("tool_calls", [])))
