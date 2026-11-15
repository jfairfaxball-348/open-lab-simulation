"""Supervisor-only recovery policy, accounting and bounded health monitoring."""

from __future__ import annotations

import json
import math
import os
import queue
import sys
import threading
import time

from . import ollama
from .ownership import process_birth

# ADR-0021: the only hosted identity a resumed hosted run may rely on. It is a
# provider-reported version string, never a cryptographic attestation.
HOSTED_IDENTITY = "provider-model-version-v1"
# The only failure after which a hosted run may restart by itself: an
# infrastructure fault inside the Laboratory, never a provider outcome.
HOSTED_RECOVERABLE_PREFIX = "the agent runtime exited unexpectedly"


def policy(config):
    value = config.get("extensions", {}).get("recovery")
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) - {
        "max_restarts", "auto_resume", "health_check_interval_s", "health_check_timeout_s",
        "hosted_identity",
    }:
        raise ValueError("invalid extensions.recovery policy")
    if "hosted_identity" in value and value["hosted_identity"] != HOSTED_IDENTITY:
        raise ValueError(f"recovery hosted_identity must be '{HOSTED_IDENTITY}'")
    if type(value.get("max_restarts")) is not int or not 0 <= value["max_restarts"] <= 100:
        raise ValueError("recovery max_restarts must be an integer from 0 to 100")
    if type(value.get("auto_resume", False)) is not bool:
        raise ValueError("recovery auto_resume must be boolean")
    result = {"auto_resume": False, "health_check_interval_s": 5,
              "health_check_timeout_s": 2, **value}
    for name in ("health_check_interval_s", "health_check_timeout_s"):
        number = result[name]
        if (
            type(number) not in (int, float) or not math.isfinite(number)
            or not 0.05 <= number <= 60
        ):
            raise ValueError(f"recovery {name} must be between 0.05 and 60 seconds")
    if "checkpointing" not in config:
        raise ValueError("recovery requires checkpointing")
    return result


def event_extension(event):
    return event.get("payload", {}).get("extensions", {})


def restarts_used(events):
    return sum(event_extension(e).get("stage") == "recovery_restart" for e in events)


def steps_used(events):
    return max(
        sum(e["type"] == "model.call" for e in events),
        sum(event_extension(e).get("stage") == "model_step_reserved" for e in events),
    )


class Accounting:
    """A crash ledger outside the worker: reserve one second ahead, renew every 0.2s.

    A crash may conservatively charge the unused part of a reservation. Downtime
    is never reconstructed from wall-clock timestamps. This is host-local state,
    not an agent-readable capability or part of the observation-event contract.
    """

    def __init__(self, archive, elapsed):
        self.archive = archive
        self.offset = elapsed
        self.started = time.monotonic()
        self.runtime = None
        self.stopped = threading.Event()
        self.error = None
        self._mutex = threading.Lock()
        self._write()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _write(self):
        with self._mutex:
            lease = time.monotonic() + 1
            target = self.archive / ".supervisor.json"
            stage = self.archive / ".supervisor.tmp"
            if target.is_symlink() or stage.is_symlink() or target.is_junction():
                raise ValueError("recovery accounting cannot be a link")
            with stage.open("w", encoding="utf-8", newline="\n") as handle:
                json.dump({"run_id": self.archive.name,
                           "elapsed_s": self.offset + lease - self.started,
                           "runtime": self.runtime}, handle)
                handle.flush()
                os.fsync(handle.fileno())
            stage.replace(target)
            self.lease_deadline = lease

    def _loop(self):
        while not self.stopped.wait(0.2):
            try:
                self._write()
            except (OSError, ValueError) as error:
                self.error = error
                return

    def attach(self, pid):
        self.runtime = {"pid": pid, "birth": process_birth(pid)}
        self._write()

    def close(self):
        self.stopped.set()
        self.thread.join()


def interrupted_elapsed(archive):
    value = json.loads((archive / ".supervisor.json").read_text(encoding="utf-8"))
    elapsed = value["elapsed_s"]
    if (
        value["run_id"] != archive.name or type(elapsed) not in (float, int)
        or not math.isfinite(elapsed) or elapsed < 0
    ):
        raise ValueError("invalid interrupted-run accounting")
    runtime = value.get("runtime")
    if runtime is not None:
        birth = process_birth(runtime["pid"])
        if birth is not None and birth == runtime["birth"]:
            raise ValueError("interrupted run still has a live runtime; refusing recovery")
        if sys.platform.startswith("linux"):
            try:
                os.killpg(runtime["pid"], 0)
            except ProcessLookupError:
                pass
            except PermissionError:
                raise ValueError("cannot verify the old runtime process group") from None
            else:
                raise ValueError("old runtime process group still exists; refusing recovery")
    return elapsed


class HealthMonitor:
    """Network checks run off the deadline thread, one outstanding probe at most."""

    def __init__(self, model, settings):
        self.model = model
        self.settings = settings
        self.results = queue.Queue()
        self.pending = False
        self.pending_since = 0
        self.next_check = 0

    def poll(self):
        try:
            result = self.results.get_nowait()
        except queue.Empty:
            result = None
            if self.pending and (
                time.monotonic() - self.pending_since > self.settings["health_check_timeout_s"]
            ):
                return "model health check timed out"
        else:
            self.pending = False
            self.next_check = time.monotonic() + self.settings["health_check_interval_s"]
        if not self.pending and time.monotonic() >= self.next_check:
            self.pending = True
            self.pending_since = time.monotonic()
            threading.Thread(target=self._probe, daemon=True).start()
        return result

    def _probe(self):
        try:
            tags = ollama._request(
                ollama.native_base(self.model["base_url"]), "/api/tags",
                timeout=self.settings["health_check_timeout_s"],
            )
            wanted = {self.model["provider_ref"], self.model["provider_ref"] + ":latest"}
            healthy = any(
                isinstance(m, dict) and wanted & {m.get("name"), m.get("model")}
                and m.get("digest") == self.model["identity"].get("digest")
                for m in tags.get("models", [])
            )
            self.results.put("healthy" if healthy else "model identity unavailable or changed")
        except (ollama.OllamaError, AttributeError, TypeError):
            self.results.put("model server health check failed")
