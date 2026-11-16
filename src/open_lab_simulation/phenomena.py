"""Fixed supervisor-owned laws, private snapshots and deterministic journal replay."""
from __future__ import annotations

import copy
import hashlib

from jsonschema import Draft202012Validator

from .contracts import canonical_json, sha256_text

VERSION = "coupled-fields-v1"
SIZE = 6
MAX_TICK = 1_000_000
IDS = ("world.observe", "world.act", "world.advance")


def invocation(ident):
    return {"interface": "mcp", "server": "supervisor-world", "tool": ident}


def policy(config, world):
    value = config.get("extensions", {}).get("phenomena")
    if value is None:
        return None
    if (not isinstance(value, dict) or set(value) != {"seed"}
            or type(value["seed"]) is not int or not 0 <= value["seed"] <= 0xFFFFFFFF):
        raise ValueError("phenomena requires one unsigned 32-bit integer seed")
    if world.manifest["environment"]["isolation"]["minimum_tier"] != "container":
        raise ValueError("phenomena requires the container-tier Habitat")
    for ident in IDS:
        if world.capabilities.get(ident, (None, {}))[1].get("invocation") != invocation(ident):
            raise ValueError("phenomena requires all three reserved world capabilities")
    return dict(value)


def digest(state):
    return sha256_text(canonical_json(state))


class Field:
    def __init__(self, seed, state=None):
        if type(seed) is not int or not 0 <= seed <= 0xFFFFFFFF:
            raise ValueError("invalid field seed")
        if state is None:
            values = [hashlib.sha256(f"{VERSION}:{seed}:{x}:{y}".encode()).digest()
                      for y in range(SIZE) for x in range(SIZE)]
            state = {"version": VERSION, "seed": seed, "tick": 0,
                     "water": [100 + 2 * v[0] for v in values],
                     "activity": [10 + v[1] % 60 for v in values], "drive": [0] * SIZE**2}
        if (not isinstance(state, dict) or set(state) != {
                "version", "seed", "tick", "water", "activity", "drive"}
                or state["version"] != VERSION or type(state["seed"]) is not int
                or state["seed"] != seed or type(state["tick"]) is not int
                or not 0 <= state["tick"] <= MAX_TICK):
            raise ValueError("invalid private field snapshot")
        for name, low, high in (("water", 0, 1000), ("activity", 0, 200), ("drive", -3, 3)):
            values = state[name]
            if (not isinstance(values, list) or len(values) != SIZE**2
                    or any(type(v) is not int or not low <= v <= high for v in values)):
                raise ValueError("invalid private field array")
        self.state = copy.deepcopy(state)

    def observe(self):
        state = self.state
        return {"available": True, "width": SIZE, "height": SIZE, "tick": state["tick"],
                "cells": [{"x": i % SIZE, "y": i // SIZE, "water": w,
                           "activity": state["activity"][i], "drive": state["drive"][i]}
                          for i, w in enumerate(state["water"])]}

    def apply(self, ident, arguments):
        if not isinstance(arguments, dict):
            raise ValueError("world arguments must be an object")
        expected = {"world.observe": set(), "world.act": {"x", "y", "drive"},
                    "world.advance": {"ticks"}}
        if ident not in expected or set(arguments) != expected[ident]:
            raise ValueError("invalid world operation fields")
        if any(type(v) is not int for v in arguments.values()):
            raise ValueError("world controls must be integers")
        if ident == "world.act":
            x, y, drive = (arguments[k] for k in ("x", "y", "drive"))
            if not (0 <= x < SIZE and 0 <= y < SIZE and -3 <= drive <= 3):
                raise ValueError("world control out of range")
            self.state["drive"][y * SIZE + x] = drive
        elif ident == "world.advance":
            ticks = arguments["ticks"]
            if not 1 <= ticks <= 100 or self.state["tick"] + ticks > MAX_TICK:
                raise ValueError("world advance exceeds tick bounds")
            for _ in range(ticks):
                self._tick()
        return self.observe()

    def _tick(self):
        state = self.state
        water, activity = state["water"], state["activity"]
        next_water, next_activity = [], []
        tick = state["tick"] + 1
        forcing = ((tick + state["seed"]) % 16) // 4 - 1
        for i, w in enumerate(water):
            x, y = i % SIZE, i // SIZE
            neighbours = (y * SIZE + (x - 1) % SIZE, y * SIZE + (x + 1) % SIZE,
                          ((y - 1) % SIZE) * SIZE + x, ((y + 1) % SIZE) * SIZE + x)
            transport = (sum(water[j] for j in neighbours) - 4 * w) // 16
            next_water.append(max(0, min(1000, w + transport + forcing
                                        + 8 * state["drive"][i] - activity[i] // 12)))
            next_activity.append(max(0, min(200, activity[i] + w // 100 - 2 - activity[i] // 20)))
        state.update(tick=tick, water=next_water, activity=next_activity)


class Service:
    def __init__(self, config, world, state=None):
        self.config, self.world = config, world
        chosen = policy(config, world)
        if chosen is None:
            raise ValueError("phenomena is not configured")
        self.field = Field(chosen["seed"], state)

    def execute(self, ident, arguments, recorder, actor, correlation):
        # Import here to avoid making the tool host the owner of this service.
        from .toolhost import permission_shortfall

        caps = self.config.get("capabilities", {})
        manifest = self.world.capabilities.get(ident, (None, {}))[1]
        if (ident not in IDS or manifest.get("invocation") != invocation(ident)
                or ident in caps.get("deny", [])
                or ("allow" in caps and ident not in caps["allow"])
                or permission_shortfall(manifest["permissions"], self.config["permissions"])):
            return {"status": "denied", "error": "world operation is not permitted"}
        if not Draft202012Validator(manifest["input_schema"]).is_valid(arguments):
            return {"status": "error", "error": "invalid world input"}
        candidate = Field(self.field.state["seed"], self.field.state)
        try:
            result = candidate.apply(ident, arguments)
        except ValueError as error:
            return {"status": "error", "error": str(error)}
        if not Draft202012Validator(manifest["output_schema"]).is_valid(result):
            raise ValueError("world service produced invalid output")
        recorder.record(recorder="supervisor", type="supervisor.action", actor=actor,
                        payload={"action": "limit_enforced", "detail": "world operation recorded",
                                 "extensions": {"stage": "phenomena_call", "version": VERSION,
                                                "correlation_id": correlation, "id": ident,
                                                "arguments": arguments, "result": result,
                                                "before": digest(self.field.state),
                                                "after": digest(candidate.state)}})
        recorder.sync()
        self.field = candidate
        return {"status": "ok", "result": result}


def replay(config, events):
    """Reconstruct state at each checkpoint and validate lawfulness and delivered results."""
    checkpoints, calls, pending, used = {}, {}, {}, set()
    chosen = config.get("extensions", {}).get("phenomena")
    stages = [e["payload"].get("extensions", {}).get("stage", "") for e in events]
    if chosen is None:
        return (["phenomena records without configuration"] if any(
            isinstance(s, str) and s.startswith("phenomena_") for s in stages) else []), checkpoints
    try:
        if not isinstance(chosen, dict) or set(chosen) != {"seed"}:
            raise ValueError("invalid replay seed configuration")
        field = Field(chosen["seed"])
        rooted, root_count, selected = False, 0, None
        for event in events:
            payload, kind = event["payload"], event["type"]
            ext = payload.get("extensions", {})
            stage = ext.get("stage")
            correlation = event.get("correlation_id")
            if kind == "recovery":
                selected = payload["checkpoint_id"]
                if selected not in checkpoints:
                    raise ValueError("recovery field checkpoint is unknown")
                field = Field(chosen["seed"], checkpoints[selected])
                rooted = False
            if kind == "tool.call":
                ident, arguments = payload["capability_id"], payload["arguments"]
                if ident in ("capabilities.invoke", "capabilities.demonstrate"):
                    calls[correlation] = (arguments.get("id"), arguments.get("arguments"),
                                          ident == "capabilities.demonstrate")
                else:
                    calls[correlation] = (ident, arguments, False)
            if stage == "phenomena_root":
                if (event["recorder"] != "supervisor" or kind != "supervisor.action"
                        or ext.get("version") != VERSION or ext.get("seed") != chosen["seed"]
                        or ext.get("checkpoint_id") != selected or rooted
                        or ext.get("state_sha256") != digest(field.state)):
                    raise ValueError("invalid phenomena root")
                rooted, root_count = True, root_count + 1
            if stage == "phenomena_call":
                cid = ext["correlation_id"]
                original = calls.get(cid)
                if (not rooted or event["recorder"] != "supervisor" or kind != "supervisor.action"
                        or ext.get("version") != VERSION or cid in used or not original
                        or original[0] != ext["id"]
                        or (not original[2] and original[1] != ext["arguments"])
                        or ext["before"] != digest(field.state)):
                    raise ValueError("invalid phenomena call binding or prior state")
                result = field.apply(ext["id"], ext["arguments"])
                if result != ext["result"] or ext["after"] != digest(field.state):
                    raise ValueError("phenomena result or state differs from replay")
                used.add(cid)
                pending[cid] = (result, ext["arguments"], original[2])
            if kind == "tool.result" and correlation in pending:
                result, arguments, demo = pending.pop(correlation)
                outcome = payload
                if demo:
                    if payload.get("result", {}).get("input") != arguments:
                        raise ValueError("world demonstration arguments differ")
                    outcome = payload.get("result", {}).get("outcome", {})
                if outcome.get("status") != "ok" or outcome.get("result") != result:
                    raise ValueError("delivered world result differs from supervisor record")
            if kind == "checkpoint.created":
                if not rooted:
                    raise ValueError("field checkpoint precedes root")
                checkpoints[payload["checkpoint_id"]] = copy.deepcopy(field.state)
        if root_count == 0:
            raise ValueError("missing phenomena root")
    except (ValueError, KeyError, TypeError) as error:
        return [f"phenomena replay: {error}"], checkpoints
    return [], checkpoints
