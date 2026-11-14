"""The agent runtime: a child process of the supervisor, holding no authority.

It is told where the workspace and the world release are, and nothing else.
It does not know where the run archive is and cannot write to it. To have
something recorded it sends a request up a pipe and waits for the supervisor
to acknowledge it, so the record is written before the agent can act again.

Pipe protocol, one JSON object per line:
  stdin   first line: the launch specification; then ``{"ack": n}`` per request
  stdout  ``{"event": {...}}`` requests, then one ``{"end": {...}}``
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from .model import ModelClient, ModelError, OpenAIChatModel, ScriptedModel
from .toolhost import ToolHost

MAX_OUTPUT_CHARS = 100_000


class Channel:
    def __init__(self) -> None:
        # Keep the protocol stream private: anything printed by accident goes to stderr.
        self._out = os.fdopen(os.dup(1), "w", encoding="utf-8", newline="\n")
        os.dup2(2, 1)
        self._in = sys.stdin.buffer

    def read(self) -> dict[str, Any]:
        line = self._in.readline()
        if not line:
            raise SystemExit(3)  # the supervisor has gone; there is nobody to work for
        return json.loads(line.decode("utf-8"))

    def emit(self, request: dict[str, Any]) -> None:
        self._out.write(json.dumps({"event": request}) + "\n")
        self._out.flush()
        self.read()

    def end(self, reason: str, steps: int) -> None:
        self._out.write(json.dumps({"end": {"reason": reason, "steps": steps}}) + "\n")
        self._out.flush()


def run_loop(
    model: ModelClient,
    host: ToolHost,
    emit,
    prompt: str,
    max_steps: int | None,
    actor: dict[str, Any],
) -> tuple[str, int]:
    """The agent loop. Returns why it ended and how many steps it took."""
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    steps = 0
    while max_steps is None or steps < max_steps:
        started = time.monotonic()
        try:
            reply = model.chat(messages)
        except ModelError as error:
            emit({
                "recorder": "agent_runtime", "type": "error", "actor": {"kind": "system"},
                "payload": {"origin": "model", "message": str(error), "recoverable": False},
            })
            return "model_error", steps
        if reply is None:
            return "model_finished", steps
        steps += 1
        emit({
            "recorder": "model_gateway", "type": "model.call", "actor": actor,
            "payload": {
                "model_id": model.model_id, "purpose": "orchestration",
                "duration_ms": int((time.monotonic() - started) * 1000),
                **reply.details,
            },
        })
        output: dict[str, Any] = {"text": reply.text[:MAX_OUTPUT_CHARS]}
        if len(reply.text) > MAX_OUTPUT_CHARS:
            output["truncated"] = True
        emit({"recorder": "agent_runtime", "type": "agent.output", "actor": actor,
              "payload": output})
        # The conversation is kept in the chat-completions shape, as the model wrote it.
        said: dict[str, Any] = {"role": "assistant", "content": reply.text}
        if reply.tool_calls:
            said["tool_calls"] = [
                {"id": call.id, "type": "function",
                 "function": {"name": call.name, "arguments": call.raw_arguments}}
                for call in reply.tool_calls
            ]
        messages.append(said)
        for call in reply.tool_calls:
            result = host.call(call.capability_id, call.arguments)
            messages.append({"role": "tool", "tool_call_id": call.id,
                             "content": json.dumps(result)})
        if not reply.tool_calls and model.ends_without_tool_calls:
            # Nobody else speaks in a run, so there is nothing for the model to answer.
            return "no_tool_calls", steps
    return "max_steps", steps


def main() -> int:
    channel = Channel()
    spec = channel.read()
    release = Path(spec["release_dir"])
    capabilities = {}
    for entry in spec["capabilities"]:
        path = release / entry["manifest_path"]
        capabilities[entry["id"]] = (path.parent, json.loads(path.read_text(encoding="utf-8")))

    chosen = spec["model"]
    model: ModelClient
    if chosen["kind"] == "scripted":
        model = ScriptedModel(chosen["replies"])
    else:
        allow, deny = spec.get("allow"), set(spec.get("deny") or [])
        offered = [
            manifest for capability_id, (_, manifest) in capabilities.items()
            if capability_id not in deny and (allow is None or capability_id in allow)
        ]
        model = OpenAIChatModel(
            chosen["model_id"], chosen["base_url"], chosen["provider_ref"], offered,
            chosen.get("sampling"), chosen.get("context_tokens"),
        )
    actor = spec["actor"]
    host = ToolHost(
        capabilities, Path(spec["workspace"]), spec["permissions"], channel.emit, actor,
        allow=spec.get("allow"), deny=spec.get("deny"),
    )
    reason, steps = run_loop(
        model, host, channel.emit, spec["initial_prompt"], spec.get("max_steps"), actor
    )
    channel.end(reason, steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
