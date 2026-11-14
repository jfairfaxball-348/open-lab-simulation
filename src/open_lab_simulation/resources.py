"""Supervisor-owned process-tree limits, workspace admission and observation."""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
import threading
import time

MIB = 1024**2


def validate(config):
    limits = config["limits"].get("resources", {})
    if "max_model_tokens" in config["limits"]:
        raise ValueError("max_model_tokens cannot yet be enforced")
    unsupported = set(limits) - {"cpu_cores", "ram_mb", "max_processes", "disk_mb"}
    if unsupported:
        raise ValueError(f"resource limit cannot be enforced: {sorted(unsupported)[0]}")
    if sys.platform != "win32" and set(limits) & {"cpu_cores", "ram_mb", "max_processes"}:
        raise ValueError("CPU/RAM/process-count limits require Windows Job Objects")
    if "cpu_cores" in limits:
        cores = os.cpu_count() or 1
        if not math.isfinite(limits["cpu_cores"]):
            raise ValueError("cpu_cores must be finite")
        rate = int(limits["cpu_cores"] * 10000 / cores)
        if cores > 64 or not 1 <= rate <= 10000:
            raise ValueError("cpu_cores cannot be represented on this host")
    if limits.get("max_processes", 1) > 0xFFFFFFFF:
        raise ValueError("max_processes cannot be represented by a Job Object")
    if limits.get("ram_mb", 1) > sys.maxsize // MIB:
        raise ValueError("ram_mb cannot be represented by a Job Object")
    sampling = config.get("extensions", {}).get("resource_sampling", {})
    if not isinstance(sampling, dict) or set(sampling) - {"interval_s"}:
        raise ValueError("invalid resource_sampling settings")
    interval = sampling.get("interval_s", 1)
    if (
        type(interval) not in (int, float) or not math.isfinite(interval)
        or not 0.05 <= interval <= 60
    ):
        raise ValueError("resource_sampling interval_s must be from 0.05 to 60 seconds")
    return limits, interval


def workspace_bytes(workspace):
    total = 0
    for path in workspace.rglob("*"):
        if path.is_symlink() or path.is_junction():
            raise ValueError("workspace quota cannot follow links")
        if path.is_file():
            try:
                total += path.stat().st_size
            except FileNotFoundError:
                pass  # a concurrent SQLite sidecar may have been removed
    return total


def parse_gpu(text):
    values = []
    for line in text.splitlines():
        used, total = [float(part.strip()) for part in line.split(",")]
        if not all(math.isfinite(v) for v in (used, total)) or not 0 <= used <= total:
            raise ValueError("invalid GPU measurement")
        values.append((used, total))
    if not values:
        raise ValueError("no GPU measurement")
    return {"used_mb": sum(v[0] for v in values), "available_mb": sum(v[1]-v[0] for v in values)}


class GpuProbe:
    def __init__(self):
        self.value = None
        self.pending = False
        self.started = 0
        self.measured = 0

    def poll(self):
        now = time.monotonic()
        if not self.pending and (not self.started or now - self.started >= 5):
            self.pending, self.started = True, now
            threading.Thread(target=self._read, daemon=True).start()
        if self.value is not None and now - self.measured < 10:
            return {**self.value, "age_s": max(0, now - self.measured)}
        return None

    def _read(self):
        try:
            command = shutil.which("nvidia-smi")
            if command is None:
                return
            done = subprocess.run(
                [command, "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2, check=True,
                creationflags=0x08000000 if sys.platform == "win32" else 0,
            )
            self.value = parse_gpu(done.stdout)
            self.measured = time.monotonic()
        except (OSError, ValueError, subprocess.SubprocessError):
            self.value = None
        finally:
            self.pending = False


GPU = GpuProbe()


class Controller:
    def __init__(self, config, workspace, world, recorder):
        self.limits, self.interval = validate(config)
        self.workspace, self.recorder = workspace, recorder
        self.writers = {ident for ident, (_, manifest) in world.capabilities.items()
                        if manifest["permissions"]["filesystem"] == "workspace_write"}
        self.next_sample = 0
        self.next_quota = 0
        self.disk_denied = False

    def action(self, action, limit, detail, stage="resource_limit"):
        self.recorder.record(recorder="supervisor", type="supervisor.action",
                             actor={"kind": "supervisor"}, payload={
                                 "action": action, "detail": detail,
                                 "extensions": {"stage": stage, "limit": limit},
                             })

    def installed(self):
        for name in self.limits:
            detail = f"configured {name}={self.limits[name]}"
            if name == "cpu_cores":
                cpus = os.cpu_count() or 1
                rate = int(self.limits[name] * 10000 / cpus)
                detail += f"; logical_processors={cpus}; job_rate={rate}"
            self.action("limit_enforced", name, detail,
                        "resource_limit_installed")

    def admit(self, capability_id):
        if capability_id not in self.writers or "disk_mb" not in self.limits:
            return None
        if self.disk_denied or workspace_bytes(self.workspace) >= self.limits["disk_mb"] * MIB:
            self.disk_denied = True
            self.action("permission_denied", "disk_mb",
                        "workspace quota denies a writing capability")
            return "workspace disk quota reached"
        return None

    def tick(self, tree, open_calls, *, force=False):
        violations = tree.violations()
        if violations:
            return violations[0]
        now = time.monotonic()
        size = None
        if "disk_mb" in self.limits and (force or now >= self.next_quota):
            self.next_quota = now + 0.1
            size = workspace_bytes(self.workspace)
            if size >= self.limits["disk_mb"] * MIB:
                if not self.disk_denied:
                    self.action("limit_enforced", "disk_mb",
                                "workspace quota reached; future writes denied")
                self.disk_denied = True
                if size > self.limits["disk_mb"] * MIB and self.writers & set(open_calls.values()):
                    return "disk_mb"
        if not force and now < self.next_sample:
            return None
        self.next_sample = now + self.interval
        if size is None:
            size = workspace_bytes(self.workspace)
        metrics = tree.sample()
        metrics["workspace_disk_mb"] = size / MIB
        gpu = GPU.poll()
        if gpu is not None:
            metrics["vram_mb"] = gpu["used_mb"]
        unavailable = [k for k in ("cpu_percent", "ram_mb", "process_count", "vram_mb")
                       if k not in metrics]
        metrics["extensions"] = {
            "unavailable": unavailable, "cpu_basis": "100_percent_per_logical_core",
            "ram_basis": "aggregate_process_private_commit", "vram_scope": "all_host_nvidia_gpus",
            **({"vram_available_mb": gpu["available_mb"], "vram_age_s": gpu["age_s"]}
               if gpu is not None else {}),
        }
        self.recorder.record(recorder="supervisor", type="resource.sample",
                             actor={"kind": "supervisor"}, payload=metrics)
        return None
