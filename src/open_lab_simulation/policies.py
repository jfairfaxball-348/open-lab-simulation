"""Versioned infrastructure treatments; never inferred from model prose."""

from __future__ import annotations

from .contracts import canonical_json, sha256_text
from .model import DEFAULT_CONTEXT_TOKENS

LEGACY = "legacy-immediate-v1"
QUIET = "quiet-until-deadline-v1"
SCHEDULED = "self-scheduled-v1"
CONTEXT = "oldest-tools-turns-v1"
MEMORY = "explicit-tools-v1"
TURN = "user-continuation-v1"


def digest(document):
    return sha256_text(canonical_json(document))


def resolve(config, continuation, context_tokens=None):
    raw = config.get("extensions", {}).get("runtime_policy", {})
    allowed = {"id", "max_tool_free_replies", "context_guard_tokens", "context_policy",
               "memory_policy", "turn_semantics", "max_sleep_s"}
    if not isinstance(raw, dict) or set(raw) - allowed:
        raise ValueError("unsupported runtime_policy fields")
    ident = raw.get("id", LEGACY)
    if ident not in (LEGACY, QUIET, SCHEDULED):
        raise ValueError("unsupported runtime policy id")
    for key, value in (("context_policy", CONTEXT), ("memory_policy", MEMORY),
                       ("turn_semantics", TURN)):
        if raw.get(key, value) != value:
            raise ValueError(f"unsupported runtime policy {key}")
    guard = raw.get("context_guard_tokens", context_tokens or DEFAULT_CONTEXT_TOKENS)
    if type(guard) is not int or guard < 1:
        raise ValueError("context_guard_tokens must be a positive integer")
    threshold = raw.get("max_tool_free_replies", 1) if ident != LEGACY else None
    if ident == LEGACY and "max_tool_free_replies" in raw:
        raise ValueError("legacy continuation has no tool-free reply threshold")
    if ident != LEGACY and (type(threshold) is not int or not 1 <= threshold <= 16):
        raise ValueError("max_tool_free_replies must be an integer from 1 through 16")
    maximum = raw.get("max_sleep_s", 3600)
    if "max_sleep_s" in raw and ident != SCHEDULED:
        raise ValueError("max_sleep_s requires self-scheduled-v1")
    if ident == SCHEDULED and (type(maximum) is not int or not 1 <= maximum <= 3600):
        raise ValueError("max_sleep_s must be an integer from 1 through 3600")
    return {
        "id": ident, "max_tool_free_replies": threshold,
        "context_guard_tokens": guard, "context_policy": CONTEXT,
        "memory_policy": MEMORY, "turn_semantics": TURN,
        "continuation_message": continuation,
        "repair_policy": "syntactic-tool-repair-v1",
        "idle_wake": "explicit-active-time-target" if ident == SCHEDULED else "none",
        "wait_budget": "remaining-active-wall-clock",
        **({"max_sleep_s": maximum} if ident == SCHEDULED else {}),
        "experiment_sha256": digest(config),
    }


def validate_state(state, policy):
    """A checkpoint can preserve a policy state, never invent new authority."""
    if not isinstance(state, dict) or set(state) != {"tool_free_replies", "idle"}:
        raise ValueError("invalid runtime policy state")
    count = state["tool_free_replies"]
    if type(count) is not int or count < 0 or type(state["idle"]) is not bool:
        raise ValueError("invalid runtime policy counters")
    if policy["id"] == LEGACY:
        if state["idle"]:
            raise ValueError("legacy policy cannot be idle")
    elif policy["id"] == SCHEDULED:
        if count >= policy["max_tool_free_replies"] and not state["idle"]:
            raise ValueError("scheduled quiet latch differs from reply counter")
    elif state["idle"] != (count >= policy["max_tool_free_replies"]):
        raise ValueError("quiet policy idle latch differs from reply counter")
