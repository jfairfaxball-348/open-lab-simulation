"""Supervisor-only active-budget snapshots and bounded self-scheduled waking."""
from __future__ import annotations

from jsonschema import Draft202012Validator

from . import policies

IDS = ("time.remaining", "time.sleep")


def invocation(ident):
    return {"interface": "mcp", "server": "supervisor-time", "tool": ident}


class Clock:
    def __init__(self, config, world, policy, elapsed, state=None):
        self.config, self.world, self.policy, self.elapsed = config, world, policy, elapsed
        self.limit_ms = config["limits"]["wall_clock_s"] * 1000
        state = state if state is not None else {"wake_at_elapsed_ms": None, "wake_ready": False}
        if (not isinstance(state, dict) or set(state) != {"wake_at_elapsed_ms", "wake_ready"}
                or type(state["wake_ready"]) is not bool):
            raise ValueError("invalid private wake state")
        target = state["wake_at_elapsed_ms"]
        if (target is not None and (type(target) is not int or not 0 < target < self.limit_ms)
                or (target is not None and state["wake_ready"])
                or (policy["id"] != policies.SCHEDULED and (target is not None
                                                          or state["wake_ready"]))):
            raise ValueError("wake state exceeds policy or original budget")
        self.state = dict(state)

    def active_ms(self):
        return max(0, min(self.limit_ms, int(self.elapsed() * 1000)))

    def snapshot(self):
        return dict(self.state)

    def pending(self):
        return self.state["wake_at_elapsed_ms"] is not None

    def _record(self, recorder, actor, **data):
        recorder.record(recorder="supervisor", type="supervisor.action", actor=actor,
                        payload={"action": "limit_enforced", "detail": "active-time service",
                                 "extensions": data})
        recorder.sync()

    def barrier(self, policy_state, recorder, actor):
        if self.state["wake_ready"] and policy_state.get("idle") is False:
            self.state["wake_ready"] = False
        target = self.state["wake_at_elapsed_ms"]
        if target is not None and self.active_ms() >= target:
            self._record(recorder, actor, stage="time_wake", wake_at_elapsed_ms=target,
                         elapsed_ms=self.active_ms())
            self.state = {"wake_at_elapsed_ms": None, "wake_ready": True}

    def execute(self, ident, arguments, recorder, actor, correlation):
        from .toolhost import permission_shortfall

        caps = self.config.get("capabilities", {})
        manifest = self.world.capabilities.get(ident, (None, {}))[1]
        if (ident not in IDS or manifest.get("invocation") != invocation(ident)
                or ident in caps.get("deny", [])
                or ("allow" in caps and ident not in caps["allow"])
                or permission_shortfall(manifest["permissions"], self.config["permissions"])):
            return {"status": "denied", "error": "time service is not permitted"}
        if not Draft202012Validator(manifest["input_schema"]).is_valid(arguments):
            return {"status": "error", "error": "invalid time service input"}
        elapsed = self.active_ms()
        result = {"available": True, "remaining_ms": self.limit_ms - elapsed,
                  "elapsed_ms": elapsed, "budget_ms": self.limit_ms}
        if ident == "time.sleep":
            seconds = arguments["seconds"]
            if self.policy["id"] != policies.SCHEDULED:
                return {"status": "denied", "error": "sleep requires self-scheduled-v1"}
            target = elapsed + seconds * 1000
            if self.pending() or seconds > self.policy["max_sleep_s"] or target >= self.limit_ms:
                return {"status": "error", "error": "sleep exceeds policy or remaining budget"}
            result.update(sleeping=True, wake_at_elapsed_ms=target)
            self._record(recorder, actor, stage="time_sleep_scheduled", correlation_id=correlation,
                         **result)
            self.state = {"wake_at_elapsed_ms": target, "wake_ready": False}
        else:
            self._record(recorder, actor, stage="time_remaining", correlation_id=correlation,
                         **result)
        return {"status": "ok", "result": result}


def verify(config, policy, events):
    """Check recorded budget arithmetic, pending sleeps and wake/inference ordering."""
    pending, checkpoints, calls, results = None, {}, {}, {}
    if (not policy or policy.get("id") != policies.SCHEDULED) and not any(
            e["payload"].get("extensions", {}).get("stage") in (
                "time_remaining", "time_sleep_scheduled", "time_wake") for e in events):
        return [], checkpoints
    limit = config["limits"]["wall_clock_s"] * 1000
    try:
        for event in events:
            kind, payload = event["type"], event["payload"]
            ext = payload.get("extensions", {})
            stage = ext.get("stage")
            if kind == "tool.call":
                target, args = payload["capability_id"], payload["arguments"]
                demo = target == "capabilities.demonstrate"
                if target == "capabilities.invoke":
                    target, args = args.get("id"), args.get("arguments")
                elif target == "capabilities.demonstrate":
                    target, args = args.get("id"), None
                calls[event["correlation_id"]] = (target, args, demo)
            if kind == "recovery":
                pending = checkpoints[payload["checkpoint_id"]]
            if stage in ("time_remaining", "time_sleep_scheduled"):
                elapsed, remaining = ext["elapsed_ms"], ext["remaining_ms"]
                expected = "time.remaining" if stage == "time_remaining" else "time.sleep"
                original = calls.get(ext["correlation_id"])
                if (event["recorder"] != "supervisor" or not original or original[0] != expected
                        or type(elapsed) is not int or type(remaining) is not int
                        or not 0 <= elapsed <= limit or remaining != limit - elapsed
                        or ext["budget_ms"] != limit or ext.get("available") is not True):
                    raise ValueError("invalid time budget observation")
                expected_result = {key: ext[key] for key in (
                    "available", "remaining_ms", "elapsed_ms", "budget_ms")}
                if stage == "time_sleep_scheduled":
                    target = ext["wake_at_elapsed_ms"]
                    if (not policy or policy["id"] != policies.SCHEDULED or pending is not None
                            or type(target) is not int or not elapsed < target < limit
                            or (target - elapsed) % 1000
                            or target - elapsed > policy["max_sleep_s"] * 1000
                            or (original[1] is not None
                                and target - elapsed != original[1]["seconds"] * 1000)):
                        raise ValueError("invalid scheduled wake target")
                    pending = target
                    expected_result.update(sleeping=True, wake_at_elapsed_ms=target)
                results[ext["correlation_id"]] = expected_result
            if kind == "tool.result" and event.get("correlation_id") in results:
                cid = event["correlation_id"]
                outcome = payload.get("result", {}).get("outcome", {}) if calls[cid][2] else payload
                if outcome.get("status") != "ok" or outcome.get("result") != results.pop(cid):
                    raise ValueError("delivered time result differs from supervisor observation")
            if stage == "time_wake":
                if (event["recorder"] != "supervisor" or pending is None
                        or ext["wake_at_elapsed_ms"] != pending or ext["elapsed_ms"] < pending):
                    raise ValueError("unbound or early scheduled wake")
                pending = None
            if stage == "model_input" and pending is not None:
                raise ValueError("model input during a pending scheduled sleep")
            if kind == "checkpoint.created":
                checkpoints[payload["checkpoint_id"]] = pending
    except (ValueError, KeyError, TypeError) as error:
        return [f"time service verification: {error}"], checkpoints
    return [], checkpoints
