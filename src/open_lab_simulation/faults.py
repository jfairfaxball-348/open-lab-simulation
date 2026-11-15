"""Infrastructure-owned, preregistered faults; none grants a privilege.

``first-permitted-execution-v1`` is the S007d benchmark fault. The S011a
reliability fault ``runtime-kill-at-active-second-v1`` has the supervisor kill
the runtime process tree once, when the run's active time reaches a registered
second, so that the run's recovery policy must restore it. It is chosen by time,
not by model behaviour, so it fires whether the model is working or idle.
"""

ID = "first-permitted-execution-v1"
OUTCOME = {"status": "error", "error": "benchmark_transient_fault",
           "extensions": {"injected": True, "fault_id": ID}}


def enabled(config):
    raw = config.get("extensions", {}).get("benchmark_fault")
    if raw is None:
        return False
    if raw != {"id": ID}:
        raise ValueError("unsupported benchmark fault configuration")
    if config.get("extensions", {}).get("benchmark", {}).get("condition") != "D":
        raise ValueError("benchmark fault requires registered condition D")
    return True


RUNTIME_KILL = "runtime-kill-at-active-second-v1"
RUNTIME_KILL_STAGE = "reliability_fault"


def runtime_kill_at(config):
    """The registered active second for the one runtime kill, or None."""
    raw = config.get("extensions", {}).get("reliability_fault")
    if raw is None:
        return None
    if not isinstance(raw, dict) or set(raw) != {"id", "at_active_s"} or raw["id"] != RUNTIME_KILL:
        raise ValueError("unsupported reliability fault configuration")
    at = raw["at_active_s"]
    if type(at) is not int or not 1 <= at < config["limits"]["wall_clock_s"]:
        raise ValueError("reliability fault at_active_s must be an integer inside the wall clock")
    settings = config.get("extensions", {}).get("recovery")
    if not isinstance(settings, dict):
        settings = {}
    restarts = settings.get("max_restarts")
    if settings.get("auto_resume") is not True or type(restarts) is not int or restarts < 1:
        raise ValueError("a reliability fault requires automatic recovery with max_restarts >= 1")
    return at
