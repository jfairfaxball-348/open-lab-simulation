"""Read the host clock as UTC; this does not report remaining run time."""
from datetime import UTC, datetime


def run(arguments):
    return {"utc": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")}
