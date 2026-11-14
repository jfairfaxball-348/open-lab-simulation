"""The container isolation tier: each capability call in a fresh rootless Podman container.

The development host is Windows; Podman runs rootless inside a WSL 2
distribution (Builder ADR-0014, ADR-0015). Everything here is invoked with
``wsl.exe --exec`` and an argument vector: ``wsl.exe --`` would pass the
arguments through a Linux shell, which rewrites them.

What a container may reach is fixed here, not by the capability or the agent:
no network, no Linux capabilities, a read-only root filesystem, a small
non-executable ``/tmp``, the verified world release read-only, and the run's
workspace as the only writable host directory (only when the capability's
filesystem permission is ``workspace_write``). The base image is pinned by
digest and is never pulled at run time.

A container outlives the ``wsl.exe`` process that started it, so killing the
runtime's process tree does not stop it. Every container carries the run's
label; the supervisor removes them all whenever a runtime segment ends, and
Podman's own ``--timeout`` is a backstop if nobody does.
"""

from __future__ import annotations

import json
import math
import secrets
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

# The image the human approved on 2026-11-14 (python:3.12-slim). Pinned by digest.
BASE_IMAGE = (
    "docker.io/library/python@sha256:"
    "05a994d2344eb364cc186910f4fc8786a38215f39f00d0acc25bff33a08f4177"
)
DISTRO = "Ubuntu"
RUN_LABEL = "open-lab.run"
# Applied when the experiment sets no ceiling of its own; recorded either way.
DEFAULT_LIMITS = {"ram_mb": 512, "cpu_cores": 1.0, "max_processes": 64}
TMPFS_MB = 64
# Container creation through WSL takes about two seconds on the development host.
STARTUP_ALLOWANCE_S = 30
CONTROL_TIMEOUT_S = 60
REQUIRED_CONTROLLERS = {"cpu", "memory", "pids"}

WORLD_MOUNT = "/world"
WORKSPACE_MOUNT = "/workspace"
_HARDENED = ("nosuid", "nodev", "noexec")


class ContainerUnavailable(Exception):
    """The container backend cannot be used as configured. Launch must be refused."""


@dataclass(frozen=True)
class Outcome:
    returncode: int
    stdout: bytes
    stderr: bytes
    timed_out: bool


def container_limits(resources: dict[str, Any] | None) -> dict[str, Any]:
    """Per-container ceilings: the experiment's own values, else the recorded defaults."""
    resources = resources or {}
    limits = {key: resources.get(key, value) for key, value in DEFAULT_LIMITS.items()}
    if not (isinstance(limits["cpu_cores"], int | float) and math.isfinite(limits["cpu_cores"])
            and limits["cpu_cores"] > 0):
        raise ContainerUnavailable("cpu_cores cannot be applied to a container")
    if limits["ram_mb"] <= TMPFS_MB:
        raise ContainerUnavailable(f"ram_mb must exceed the {TMPFS_MB} MB in-memory /tmp")
    return limits


class Backend:
    """Rootless Podman in one WSL 2 distribution, controlled from Windows."""

    def __init__(self, distro: str = DISTRO, image: str = BASE_IMAGE):
        self.distro = distro
        self.image = image
        self._paths: dict[str, str] = {}

    def _wsl(self, *argv: str) -> list[str]:
        return ["wsl.exe", "-d", self.distro, "--exec", *argv]

    def _control(self, *argv: str, timeout: float = CONTROL_TIMEOUT_S) -> bytes:
        try:
            done = subprocess.run(  # noqa: S603 - argument vector, never a shell
                self._wsl(*argv), capture_output=True, timeout=timeout, check=False
            )
        except subprocess.TimeoutExpired:
            raise ContainerUnavailable(f"'{argv[0]} {argv[1]}' did not answer") from None
        except OSError as error:
            raise ContainerUnavailable(f"wsl.exe cannot be started: {error.strerror}") from None
        if done.returncode != 0:
            # wsl.exe writes its own errors as UTF-16; neither form is trusted for detail.
            text = done.stderr.replace(b"\0", b"").decode("utf-8", errors="replace").strip()
            last = text.splitlines()[-1] if text else f"exit status {done.returncode}"
            raise ContainerUnavailable(f"'{argv[0]} {argv[1]}' failed: {last[:200]}")
        return done.stdout

    def probe(self) -> dict[str, Any]:
        """Check every property the tier relies on, and describe the backend for the archive."""
        if sys.platform != "win32":
            raise ContainerUnavailable("the container tier is implemented for Windows hosts only")
        try:
            info = json.loads(self._control("podman", "info", "--format", "json"))
            host = info["host"]
            version = info["version"]["Version"]
            rootless = host["security"]["rootless"]
            cgroups = host["cgroupVersion"]
            controllers = set(host.get("cgroupControllers") or [])
            runtime = host["ociRuntime"]["name"]
            kernel = host.get("kernel", "")
        except (ValueError, KeyError, TypeError):
            raise ContainerUnavailable("podman info is not in the expected form") from None
        if rootless is not True:
            raise ContainerUnavailable("Podman must run rootless")
        if cgroups != "v2" or not REQUIRED_CONTROLLERS <= controllers:
            raise ContainerUnavailable(
                "Podman needs cgroups v2 with the cpu, memory and pids controllers delegated"
            )
        image_digest = self.image.rpartition("@")[2]
        try:
            described = json.loads(self._control("podman", "image", "inspect", self.image))[0]
        except ContainerUnavailable:
            raise ContainerUnavailable(
                "the pinned base image is not present; it is never pulled at run time"
            ) from None
        except (ValueError, IndexError):
            raise ContainerUnavailable("podman image inspect is not in the expected form") from None
        if described.get("Digest") != image_digest:
            raise ContainerUnavailable("the local base image does not match its pinned digest")
        return {
            "tier": "container",
            "host": "windows-wsl2",
            "engine": "podman",
            "engine_version": version,
            "rootless": True,
            "cgroups": cgroups,
            "oci_runtime": runtime,
            "kernel": kernel,
            "image": self.image,
            "image_id": described.get("Id"),
        }

    def host_path(self, path: Path) -> str:
        """The WSL path of a Windows path, as WSL's own automount configuration maps it."""
        key = str(path)
        if key not in self._paths:
            # Forward slashes: wslpath reads a backslash path only through a shell.
            converted = self._control("wslpath", "-a", path.as_posix()).decode("utf-8").strip()
            if not converted.startswith("/") or "\n" in converted:
                raise ContainerUnavailable("wslpath returned an unusable path")
            self._paths[key] = converted
        return self._paths[key]

    def command(
        self,
        *,
        name: str,
        run_id: str,
        world: Path,
        capability_dir: Path,
        workspace: Path | None,
        writable: bool,
        argv: list[str],
        env: dict[str, str],
        limits: dict[str, Any],
        timeout_s: int,
    ) -> list[str]:
        relative = capability_dir.relative_to(world).as_posix()
        hardened = ",".join(_HARDENED)
        vector = [
            "podman", "run", "--rm", "--interactive", "--pull=never",
            "--name", name, "--label", f"{RUN_LABEL}={run_id}",
            "--network=none", "--read-only", "--cap-drop=ALL",
            "--security-opt=no-new-privileges", "--userns=keep-id", "--hostname=sandbox",
            f"--memory={limits['ram_mb']}m", f"--memory-swap={limits['ram_mb']}m",
            f"--cpus={limits['cpu_cores']}", f"--pids-limit={limits['max_processes']}",
            f"--timeout={timeout_s}",
            f"--tmpfs=/tmp:rw,{hardened},size={TMPFS_MB}m",
            "--volume", f"{self.host_path(world)}:{WORLD_MOUNT}:ro,{hardened}",
        ]
        if workspace is not None:
            mode = "rw" if writable else "ro"
            source = self.host_path(workspace)
            vector += ["--volume", f"{source}:{WORKSPACE_MOUNT}:{mode},{hardened}"]
        for key, value in sorted(env.items()):
            vector += ["--env", f"{key}={value}"]
        vector += ["--workdir", str(PurePosixPath(WORLD_MOUNT) / relative), self.image, *argv]
        return self._wsl(*vector)

    def run(self, vector: list[str], name: str, stdin: bytes, timeout_s: int,
            environment: dict[str, str]) -> Outcome:
        """Run one container to completion, removing it if it is still there afterwards."""
        started = time.monotonic()
        try:
            done = subprocess.run(  # noqa: S603 - argument vector, never a shell
                vector, input=stdin, capture_output=True, env=environment,
                timeout=timeout_s + STARTUP_ALLOWANCE_S, check=False,
            )
        except subprocess.TimeoutExpired:
            self.remove(name)
            return Outcome(-1, b"", b"", True)
        # Podman's --timeout stops the container without a distinctive exit status.
        timed_out = done.returncode != 0 and time.monotonic() - started >= timeout_s
        if timed_out:
            self.remove(name)
        return Outcome(done.returncode, done.stdout, done.stderr, timed_out)

    def remove(self, name: str) -> None:
        try:
            self._control("podman", "rm", "--force", "--ignore", "--time", "0", name)
        except ContainerUnavailable:
            pass  # the supervisor's sweep for the run label follows every segment

    def running(self, run_id: str) -> list[str]:
        listed = self._control(
            "podman", "ps", "--all", "--filter", f"label={RUN_LABEL}={run_id}",
            "--format", "{{.Names}}",
        )
        return [line for line in listed.decode("utf-8").split() if line]

    def sweep(self, run_id: str) -> list[str]:
        """Remove every container carrying the run's label; return the names that existed."""
        found = self.running(run_id)
        if found:
            self._control("podman", "rm", "--force", "--ignore", "--time", "0", *found)
        left = self.running(run_id)
        if left:
            raise ContainerUnavailable(f"{len(left)} container(s) of the run could not be removed")
        return found


def container_name(run_id: str) -> str:
    return f"ol-{run_id}-{secrets.token_hex(4)}"
