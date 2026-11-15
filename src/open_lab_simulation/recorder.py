"""The single writer of a run's event log.

Exactly one ``Recorder`` exists per run and it lives in the supervisor
process. The agent runtime has no handle on it or on the file: it can only
ask, over a pipe, for an observation to be recorded, and the supervisor
decides whether the request is one the runtime is entitled to make.

Each line of ``events.jsonl`` is the event's canonical JSON. From the second
event on, ``prev_event_sha256`` is the SHA-256 of the previous line (without
its newline), so any later edit, removal or reordering breaks the chain.
"""

from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .contracts import Contracts, canonical_json, jsonl_lines, sha256_text, validate_event_stream


class RecordError(Exception):
    """An event that does not satisfy the observation-event contract was refused."""


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


class Recorder:
    def __init__(
        self,
        path: Path,
        contracts: Contracts,
        experiment_id: str,
        run_id: str,
        *,
        append: bool = False,
        elapsed_s: float = 0,
    ):
        self.path = path
        self._contracts = contracts
        self._identity = {"experiment_id": experiment_id, "run_id": run_id}
        self._schema_version = contracts.version("observation-event")
        self._started = time.monotonic() - elapsed_s
        self._previous_line: str | None = None
        self.count = 0
        self.by_type: dict[str, int] = {}
        if append:
            raw = path.read_bytes()
            if not raw.endswith(b"\n"):
                raise RecordError("refusing to append to a partial event log")
            lines = jsonl_lines(raw.decode("utf-8"))
            if validate_event_stream(contracts, lines):
                raise RecordError("refusing to append to an invalid event log")
            for line in lines:
                event = json.loads(line)
                if any(event[k] != v for k, v in self._identity.items()):
                    raise RecordError("event log identity differs")
                self.count += 1
                self.by_type[event["type"]] = self.by_type.get(event["type"], 0) + 1
            self._previous_line = lines[-1]
            self._started = time.monotonic() - max(elapsed_s, event.get("monotonic_ms", 0) / 1000)
        # "x": the log must not already exist. A run never appends to another run's record.
        self._handle = path.open("a" if append else "x", encoding="utf-8", newline="\n")

    @property
    def experiment_id(self):
        return self._identity["experiment_id"]

    @property
    def run_id(self):
        return self._identity["run_id"]

    @property
    def last_event_sha256(self) -> str | None:
        return None if self._previous_line is None else sha256_text(self._previous_line)

    def record(
        self,
        *,
        recorder: str,
        type: str,  # noqa: A002 - the contract's field name
        payload: dict[str, Any],
        actor: dict[str, Any] | None = None,
        epistemic_status: str = "observed",
        correlation_id: str | None = None,
    ) -> dict[str, Any]:
        event: dict[str, Any] = {
            "schema": "open-lab/observation-event",
            "schema_version": self._schema_version,
            "event_id": f"evt-{self.count:06d}",
            **self._identity,
            "seq": self.count,
            "timestamp": utc_now(),
            "monotonic_ms": int((time.monotonic() - self._started) * 1000),
            "recorder": recorder,
            "type": type,
            "epistemic_status": epistemic_status,
            "payload": payload,
        }
        if actor is not None:
            event["actor"] = actor
        if correlation_id is not None:
            event["correlation_id"] = correlation_id
        if self._previous_line is not None:
            event["prev_event_sha256"] = sha256_text(self._previous_line)

        problems = self._contracts.validate(event, "observation-event")
        if problems:
            raise RecordError(problems[0])
        line = canonical_json(event)
        self._handle.write(line + "\n")
        self._handle.flush()
        self._previous_line = line
        self.count += 1
        self.by_type[type] = self.by_type.get(type, 0) + 1
        return event

    def sync(self) -> None:
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()
