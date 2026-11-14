"""One infrastructure-owned, preregistered benchmark fault; no new privilege."""

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
