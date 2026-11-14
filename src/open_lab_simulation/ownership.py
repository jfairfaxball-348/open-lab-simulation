"""OS-serialized archive ownership with PID birth identity, never PID alone."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import secrets
import sys
from contextlib import contextmanager
from pathlib import Path


class OwnershipError(ValueError):
    pass


def host_identity() -> str:
    try:
        if sys.platform == "win32":
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Cryptography",
                                access=winreg.KEY_READ | winreg.KEY_WOW64_64KEY) as key:
                identity = winreg.QueryValueEx(key, "MachineGuid")[0]
        else:
            identity = Path("/etc/machine-id").read_text().strip()
        if not isinstance(identity, str) or not identity:
            raise ValueError
        return hashlib.sha256(identity.encode()).hexdigest()
    except (OSError, ValueError):
        raise OwnershipError("cannot establish local host identity") from None


def process_birth(pid: int) -> str | None:
    """None means confirmed absent; inability to inspect is an error."""
    if type(pid) is not int or pid <= 0:
        raise OwnershipError("invalid writer PID")
    if sys.platform == "win32":
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetProcessTimes.argtypes = [wintypes.HANDLE, *([ctypes.c_void_p] * 4)]
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.c_void_p]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            if ctypes.get_last_error() == 87:  # ERROR_INVALID_PARAMETER: no such PID
                return None
            raise OwnershipError("cannot inspect writer process")
        try:
            code = wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
                raise OwnershipError("cannot inspect writer process state")
            if code.value != 259:  # STILL_ACTIVE
                return None
            times = [ctypes.c_uint64() for _ in range(4)]
            if not kernel.GetProcessTimes(handle, *(ctypes.byref(t) for t in times)):
                raise OwnershipError("cannot inspect writer process birth")
            return str(times[0].value)
        finally:
            kernel.CloseHandle(handle)
    if sys.platform.startswith("linux"):
        try:
            stat = Path(f"/proc/{pid}/stat").read_text()
        except FileNotFoundError:
            return None
        except OSError:
            raise OwnershipError("cannot inspect writer process birth") from None
        try:
            fields = stat[stat.rindex(")") + 2:].split()
            if fields[0] == "Z":
                return None
            boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
            return f"{boot}:{fields[19]}"
        except (OSError, ValueError, IndexError):
            raise OwnershipError("cannot inspect writer process birth") from None
    raise OwnershipError("writer recovery requires Windows or Linux process identity")


@contextmanager
def archive_lock(archive: Path, *, recover_stale: bool = False):
    # The guard inode is permanent. Unlinking an OS-locked file would allow a
    # second inode/writer to bypass the lock on POSIX.
    guard = archive / ".writer.guard"
    marker = archive / ".writer.lock"
    for path in (archive, guard, marker):
        if path.is_symlink() or path.is_junction():
            raise OwnershipError("writer lock paths cannot be links")
    handle = guard.open("a+b")
    acquired = False
    try:
        if guard.stat().st_size == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if sys.platform == "win32":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise OwnershipError("run already has a live writer lock") from None
        acquired = True
        stale = None
        if marker.exists():
            try:
                owner = json.loads(marker.read_text(encoding="utf-8"))
                if (
                    not isinstance(owner, dict) or owner["host"] != host_identity()
                    or not isinstance(owner["birth"], str) or not owner["birth"]
                ):
                    raise ValueError
                birth = process_birth(owner["pid"])
            except (OSError, ValueError, KeyError, TypeError):
                raise OwnershipError("writer lock owner cannot be verified") from None
            if birth == owner["birth"]:
                raise OwnershipError("run already has a live writer lock owner")
            if not recover_stale:
                raise OwnershipError("stale writer lock requires interrupted-run recovery")
            # Keep the evidence, while still holding the OS guard.
            stale = f"writer-{secrets.token_hex(8)}.json"
            directory = archive / "segments"
            if directory.is_symlink() or directory.is_junction():
                raise OwnershipError("writer history cannot be a link")
            directory.mkdir(exist_ok=True)
            marker.replace(directory / stale)
        owner = {"pid": os.getpid(), "birth": process_birth(os.getpid()), "host": host_identity()}
        marker.write_text(json.dumps(owner) + "\n", encoding="utf-8")
        try:
            yield stale
        finally:
            marker.unlink(missing_ok=True)
    finally:
        if acquired:
            handle.seek(0)
            if sys.platform == "win32":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()
