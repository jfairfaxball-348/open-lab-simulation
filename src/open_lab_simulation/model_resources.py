"""Live host headroom for local model admission, never a sandbox permission."""

import ctypes
import os
import shutil
import subprocess
import sys


def capacity(require_gpu=True):
    if sys.platform == "win32":
        class Memory(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong) for name in
                ("total", "available", "page_total", "page_free", "virtual_total",
                 "virtual_free", "extended")]

        memory = Memory()
        memory.length = ctypes.sizeof(memory)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
            raise ValueError("host RAM measurement unavailable")
        available = memory.available / 1024**2
    else:
        available = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1024**2
    free = 0
    if require_gpu:
        command = shutil.which("nvidia-smi")
        if command is None:
            raise ValueError("NVIDIA VRAM measurement unavailable")
        result = subprocess.run(  # noqa: S603 - fixed operator-owned probe
            [command, "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2, check=True,
            **({"creationflags": 0x08000000} if sys.platform == "win32" else {}))
        values = result.stdout.strip().splitlines()
        if len(values) != 1:
            raise ValueError("local admission requires one known NVIDIA device")
        free = float(values[0])
        if not 0 <= free < 1_000_000:
            raise ValueError("invalid VRAM measurement")
    return {"ram_available_mb": available, "vram_available_mb": free}
