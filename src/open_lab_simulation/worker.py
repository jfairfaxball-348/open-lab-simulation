"""The agent runtime: a child process of the supervisor, holding no authority.

It is told where the workspace and the world release are, and nothing else.
It does not know where the run archive is and cannot write to it. To have
something recorded it sends a request up a pipe and waits for the supervisor
to acknowledge it, so the record is written before the agent can act again.

Pipe protocol, one JSON object per line:
  stdin   first line: the launch specification; then ``{"ack": true, "seq": n}``
          per request, ``seq`` being the recorded event's number (null if refused)
  stdout  ``{"event": {...}}`` requests, then one ``{"end": {...}}``
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from . import policies
from .container import Backend
from .model import ModelClient, ModelError, OpenAIChatModel, ScriptedModel
from .registry import META_TOOLS, PREAMBLE, Registry
from .toolhost import Sandbox, ToolHost

MAX_OUTPUT_CHARS = 100_000


class Channel:
    def __init__(self) -> None:
        # Keep the protocol stream private: anything printed by accident goes to stderr.
        self._out = os.fdopen(os.dup(1), "w", encoding="utf-8", newline="\n")
        os.dup2(2, 1)
        self._in = sys.stdin.buffer
        self.denial = None

    def read(self) -> dict[str, Any]:
        line = self._in.readline()
        if not line:
            raise SystemExit(3)  # the supervisor has gone; there is nobody to work for
        return json.loads(line.decode("utf-8"))

    def emit(self, request: dict[str, Any]) -> int | None:
        """Have an event recorded; returns its sequence number once it is on record."""
        self._out.write(json.dumps({"event": request}) + "\n")
        self._out.flush()
        acknowledgement = self.read()
        self.denial = acknowledgement.get("deny")
        seq = acknowledgement.get("seq")
        return seq if isinstance(seq, int) else None

    def before_execute(self, ident, arguments):
        self._out.write(json.dumps({"benchmark_fault": {
            "capability_id": ident, "arguments_sha256": policies.digest(arguments),
        }}) + "\n")
        self._out.flush()
        reply = self.read()
        if type(reply.get("inject_fault")) is not bool:
            raise RuntimeError("execution admission was not acknowledged")
        return reply["inject_fault"]

    def model_input(self, payload):
        self._out.write(json.dumps({"model_input": payload}) + "\n")
        self._out.flush()
        if not isinstance(self.read().get("seq"), int):
            raise ModelError("model input was not acknowledged; request not sent")

    def end(self, reason: str, steps: int) -> None:
        self._out.write(json.dumps({"end": {"reason": reason, "steps": steps}}) + "\n")
        self._out.flush()

    def state(self, state: dict, final: bool = False) -> bool:
        self._out.write(json.dumps({"state": state, "final": final}) + "\n")
        self._out.flush()
        return bool(self.read().get("stop"))

    def reserve_step(self):
        self._out.write(json.dumps({"step_start": True}) + "\n")
        self._out.flush()
        self.read()


def run_loop(
    model: ModelClient,
    host: ToolHost,
    emit,
    prompt: str,
    max_steps: int | None,
    actor: dict[str, Any],
    continuation: str,
    runtime_state: dict | None = None,
    checkpoint=None,
    reserve_step=None,
    policy=None,
    record_input=None,
) -> tuple[str, int]:
    """The agent loop. Returns why it ended and how many steps it took.

    A reply without a tool call does not end the run (decision OD-5). Nobody
    else speaks in a run, so the runtime answers with the experiment's fixed
    continuation message, records exactly that, and asks the model again. The
    run then ends only at the step limit, the deadline or a failure.
    """
    if runtime_state is None:
        emit(
            {
                "recorder": "agent_runtime",
                "type": "environment.message",
                "actor": {"kind": "system"},
                "payload": {"kind": "preamble", "text": PREAMBLE},
            }
        )
    messages = (
        runtime_state["messages"]
        if runtime_state
        else [{"role": "system", "content": PREAMBLE}, {"role": "user", "content": prompt}]
    )
    steps = runtime_state["steps"] if runtime_state else 0
    tokens_used = runtime_state["tokens_used"] if runtime_state else 0
    policy = policy or {"id": policies.LEGACY}
    scheduling = dict(runtime_state.get("policy_state", {})) if runtime_state else {}
    if not scheduling:
        scheduling = {"tool_free_replies": 0, "idle": False}

    def save(final=False):
        return checkpoint and checkpoint(
            {
                "messages": messages,
                "steps": steps,
                "script_cursor": getattr(model, "position", steps),
                "call_count": host._calls,
                "tokens_used": tokens_used,
                "policy_sha256": policies.digest(policy), "policy_state": scheduling,
            },
            final,
        )

    if save():
        return "operator_pause", steps
    while max_steps is None or steps < max_steps:
        if scheduling["idle"]:
            # No inference or injected text; the supervisor retains deadline authority.
            time.sleep(0.25)
            if save():
                return "operator_pause", steps
            continue
        if reserve_step:
            reserve_step()
        started = time.monotonic()
        try:
            if record_input and not getattr(model, "records_inputs", False):
                record_input({"request": {"messages": messages}, "backend": "scripted",
                              "attempt": 1, "context": {"applied": False}})
            reply = model.chat(messages)
        except ModelError as error:
            emit(
                {
                    "recorder": "agent_runtime",
                    "type": "error",
                    "actor": {"kind": "system"},
                    "payload": {"origin": "model", "message": str(error), "recoverable": False},
                }
            )
            save(final=True)
            return "model_error", steps
        if reply is None:
            save(final=True)
            return "model_finished", steps
        steps += 1
        tokens_used += reply.details.get("input_tokens", 0) + reply.details.get("output_tokens", 0)
        emit(
            {
                "recorder": "model_gateway",
                "type": "model.call",
                "actor": actor,
                "payload": {
                    "model_id": model.model_id,
                    "purpose": "orchestration",
                    "duration_ms": int((time.monotonic() - started) * 1000),
                    **reply.details,
                },
            }
        )
        output: dict[str, Any] = {"text": reply.text[:MAX_OUTPUT_CHARS]}
        if reply.tool_calls:
            # Keep the emitted names/argument strings as evidence distinct
            # from the repaired arguments subsequently sent to the tool host.
            output["extensions"] = {
                "raw_tool_calls": [
                    {"id": call.id, "name": call.name, "arguments": call.raw_arguments}
                    for call in reply.tool_calls
                ]
            }
        if len(reply.text) > MAX_OUTPUT_CHARS:
            output["truncated"] = True
        output_seq = emit(
            {"recorder": "agent_runtime", "type": "agent.output", "actor": actor, "payload": output}
        )
        # The conversation is kept in the chat-completions shape, as the model wrote it.
        said: dict[str, Any] = {"role": "assistant", "content": reply.text}
        if reply.tool_calls:
            said["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.raw_arguments},
                }
                for call in reply.tool_calls
            ]
        messages.append(said)
        for call in reply.tool_calls:
            result = host.call(call.capability_id, call.arguments)
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)}
            )
        scheduling["tool_free_replies"] = (
            0 if reply.tool_calls else scheduling["tool_free_replies"] + 1
        )
        scheduling["idle"] = (policy["id"] == policies.QUIET and
                              scheduling["tool_free_replies"] >= policy["max_tool_free_replies"])
        if not reply.tool_calls and not scheduling["idle"]:
            payload: dict[str, Any] = {"kind": "continuation", "text": continuation}
            if output_seq is not None:
                payload["after_seq"] = output_seq
            # Recorded before the model can see it, like everything else it is told.
            emit(
                {
                    "recorder": "agent_runtime",
                    "type": "environment.message",
                    "actor": {"kind": "system"},
                    "payload": payload,
                }
            )
            messages.append({"role": "user", "content": continuation})
        if save():
            return "operator_pause", steps
    save(final=True)
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
        model = ScriptedModel(
            chosen["replies"], spec.get("runtime_state", {}).get("script_cursor", 0)
        )
    else:
        model = OpenAIChatModel(
            chosen["model_id"],
            chosen["base_url"],
            chosen["provider_ref"],
            META_TOOLS,
            chosen.get("sampling"),
            chosen["runtime_policy"]["context_guard_tokens"],
            record_input=channel.model_input,
            auth_token=chosen.get("auth_token"),
        )
    actor = spec["actor"]
    isolation = spec.get("isolation", {"tier": "process"})
    sandbox = None
    if isolation["tier"] == "container":
        sandbox = Sandbox(Backend(isolation["distro"], isolation["image"]), release,
                          isolation["run_id"], isolation["limits"])
    host = ToolHost(
        capabilities,
        Path(spec["workspace"]),
        spec["permissions"],
        channel.emit,
        actor,
        allow=spec.get("allow"),
        deny=spec.get("deny"),
        registry=Registry(release, capabilities),
        admission=lambda: channel.denial,
        sandbox=sandbox,
        before_execute=channel.before_execute if spec.get("benchmark_fault") else None,
    )
    host._calls = spec.get("runtime_state", {}).get("call_count", 0)
    reason, steps = run_loop(
        model,
        host,
        channel.emit,
        spec["initial_prompt"],
        spec.get("max_steps"),
        actor,
        spec["continuation_message"],
        runtime_state=spec.get("runtime_state"),
        checkpoint=(channel.state if spec.get("checkpointing") is not None
                    or chosen["runtime_policy"]["id"] == policies.QUIET else None),
        reserve_step=channel.reserve_step if spec.get("reserve_steps") else None,
        policy=chosen["runtime_policy"], record_input=channel.model_input,
    )
    channel.end(reason, steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
