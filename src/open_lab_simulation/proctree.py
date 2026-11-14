"""Start a process so that it and everything it spawns can be killed together.

Killing only the runtime would leave a capability it started still running,
outside any deadline. On Windows the child is placed in a Job Object, which
the kernel tracks through every descendant and which also kills the tree if
the supervisor itself dies. On POSIX the child leads its own process group.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from typing import Any

if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
    _EXTENDED_LIMIT_INFORMATION = 9
    _BASIC_ACCOUNTING_INFORMATION = 1

    class _BasicLimits(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _IoCounters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_uint64) for name in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
        )]

    class _ExtendedLimits(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _BasicLimits),
            ("IoInfo", _IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    class _Accounting(ctypes.Structure):
        _fields_ = [
            ("TotalUserTime", ctypes.c_int64),
            ("TotalKernelTime", ctypes.c_int64),
            ("ThisPeriodTotalUserTime", ctypes.c_int64),
            ("ThisPeriodTotalKernelTime", ctypes.c_int64),
            ("TotalPageFaultCount", wintypes.DWORD),
            ("TotalProcesses", wintypes.DWORD),
            ("ActiveProcesses", wintypes.DWORD),
            ("TotalTerminatedProcesses", wintypes.DWORD),
        ]

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    _kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
    _kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
    _kernel32.QueryInformationJobObject.argtypes = [
        wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD, wintypes.LPVOID]
    _kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    _kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    _kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

    class _CpuRate(ctypes.Structure):
        _fields_ = [("ControlFlags", wintypes.DWORD), ("CpuRate", wintypes.DWORD)]

    class _Port(ctypes.Structure):
        _fields_ = [("CompletionKey", ctypes.c_void_p), ("CompletionPort", wintypes.HANDLE)]

    class _ThreadEntry(ctypes.Structure):
        _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
                    ("th32ThreadID", wintypes.DWORD), ("th32OwnerProcessID", wintypes.DWORD),
                    ("tpBasePri", wintypes.LONG), ("tpDeltaPri", wintypes.LONG),
                    ("dwFlags", wintypes.DWORD)]

    class _Memory(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in (
                "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                "PagefileUsage", "PeakPagefileUsage", "PrivateUsage",
            )]

    _kernel32.CreateIoCompletionPort.argtypes = [
        wintypes.HANDLE, wintypes.HANDLE, ctypes.c_size_t, wintypes.DWORD]
    _kernel32.CreateIoCompletionPort.restype = wintypes.HANDLE
    _kernel32.GetQueuedCompletionStatus.argtypes = [
        wintypes.HANDLE, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD]
    _kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    _kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    _kernel32.Thread32First.argtypes = [wintypes.HANDLE, ctypes.c_void_p]
    _kernel32.Thread32Next.argtypes = [wintypes.HANDLE, ctypes.c_void_p]
    _kernel32.OpenThread.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    _kernel32.OpenThread.restype = wintypes.HANDLE
    _kernel32.ResumeThread.argtypes = [wintypes.HANDLE]
    _kernel32.ResumeThread.restype = wintypes.DWORD
    _kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    _kernel32.OpenProcess.restype = wintypes.HANDLE
    _kernel32.IsProcessInJob.argtypes = [wintypes.HANDLE, wintypes.HANDLE, ctypes.c_void_p]
    _psapi = ctypes.WinDLL("psapi", use_last_error=True)
    _psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]


def _resume_suspended(pid):
    snapshot = _kernel32.CreateToolhelp32Snapshot(4, 0)
    if snapshot == ctypes.c_void_p(-1).value:
        raise OSError(ctypes.get_last_error(), "thread enumeration failed")
    try:
        entry = _ThreadEntry()
        entry.dwSize = ctypes.sizeof(entry)
        more = _kernel32.Thread32First(snapshot, ctypes.byref(entry))
        while more:
            if entry.th32OwnerProcessID == pid:
                thread = _kernel32.OpenThread(2, False, entry.th32ThreadID)
                if not thread:
                    raise OSError(ctypes.get_last_error(), "OpenThread failed")
                try:
                    if _kernel32.ResumeThread(thread) == 0xFFFFFFFF:
                        raise OSError(ctypes.get_last_error(), "ResumeThread failed")
                finally:
                    _kernel32.CloseHandle(thread)
                return
            more = _kernel32.Thread32Next(snapshot, ctypes.byref(entry))
        raise OSError("suspended process has no primary thread")
    finally:
        _kernel32.CloseHandle(snapshot)


class ProcessTree:
    """A child process plus all of its descendants, killable as one."""

    def __init__(self, command: list[str], *, resources=None, **popen_arguments: Any):
        self._job = None
        self._port = None
        self._last_sample = None
        resources = resources or {}
        if sys.platform == "win32":
            job = _kernel32.CreateJobObjectW(None, None)
            if not job:
                raise OSError(ctypes.get_last_error(), "CreateJobObject failed")
            limits = _ExtendedLimits()
            limits.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if "ram_mb" in resources:
                limits.BasicLimitInformation.LimitFlags |= 0x200
                limits.JobMemoryLimit = resources["ram_mb"] * 1024**2
            if "max_processes" in resources:
                limits.BasicLimitInformation.LimitFlags |= 8
                limits.BasicLimitInformation.ActiveProcessLimit = resources["max_processes"]
            if not _kernel32.SetInformationJobObject(
                job, _EXTENDED_LIMIT_INFORMATION, ctypes.byref(limits), ctypes.sizeof(limits)
            ):
                error = ctypes.get_last_error()
                _kernel32.CloseHandle(job)
                raise OSError(error, "SetInformationJobObject failed")
            self._job = job
            try:
                if "cpu_cores" in resources:
                    rate = int(resources["cpu_cores"] * 10000 / (os.cpu_count() or 1))
                    if not 1 <= rate <= 10000:
                        raise ValueError("cpu_cores cannot be represented by this host's CPU cap")
                    cpu = _CpuRate(5, rate)  # ENABLE | HARD_CAP
                    if not _kernel32.SetInformationJobObject(
                        job, 15, ctypes.byref(cpu), ctypes.sizeof(cpu)
                    ):
                        raise OSError(ctypes.get_last_error(), "CPU hard cap unavailable")
                port = _kernel32.CreateIoCompletionPort(ctypes.c_void_p(-1), None, 0, 1)
                if not port:
                    raise OSError(ctypes.get_last_error(), "completion port unavailable")
                self._port = port
                association = _Port(None, port)
                if not _kernel32.SetInformationJobObject(
                    job, 7, ctypes.byref(association), ctypes.sizeof(association)
                ):
                    raise OSError(ctypes.get_last_error(), "job notification association failed")
                flags = popen_arguments.pop("creationflags", 0) | 4  # CREATE_SUSPENDED
                self.process = subprocess.Popen(command, creationflags=flags, **popen_arguments)
                # Assign before even the venv redirector can spawn a child.
                if not _kernel32.AssignProcessToJobObject(job, int(self.process._handle)):
                    raise OSError(ctypes.get_last_error(), "AssignProcessToJobObject failed")
                _resume_suspended(self.process.pid)
            except BaseException:
                if hasattr(self, "process"):
                    self.process.kill()
                    self.process.wait(timeout=10)
                self.close()
                raise
        else:
            self.process = subprocess.Popen(  # noqa: S603
                command, start_new_session=True, **popen_arguments
            )

    def active_processes(self) -> int:
        """How many processes of the tree are alive (best effort on POSIX)."""
        if self._job is not None:
            info = _Accounting()
            _kernel32.QueryInformationJobObject(
                self._job, _BASIC_ACCOUNTING_INFORMATION,
                ctypes.byref(info), ctypes.sizeof(info), None,
            )
            return int(info.ActiveProcesses)
        try:
            os.killpg(self.process.pid, 0)
        except (ProcessLookupError, PermissionError):
            return 0
        return 1

    def violations(self):
        result = []
        if self._port is not None:
            while True:
                message, key, pid = wintypes.DWORD(), ctypes.c_size_t(), ctypes.c_void_p()
                if not _kernel32.GetQueuedCompletionStatus(
                    self._port, ctypes.byref(message), ctypes.byref(key), ctypes.byref(pid), 0
                ):
                    if ctypes.get_last_error() != 258:  # WAIT_TIMEOUT
                        raise OSError(ctypes.get_last_error(), "job notification read failed")
                    break
                if message.value in (3, 9, 10):
                    result.append("max_processes" if message.value == 3 else "ram_mb")
        return result

    def sample(self):
        if self._job is None:
            return {}
        info = _Accounting()
        if not _kernel32.QueryInformationJobObject(
            self._job, 1, ctypes.byref(info), ctypes.sizeof(info), None
        ):
            raise OSError(ctypes.get_last_error(), "job accounting unavailable")
        result = {"process_count": int(info.ActiveProcesses)}
        now = time.monotonic()
        cpu_s = (info.TotalUserTime + info.TotalKernelTime) / 10_000_000
        if self._last_sample is not None:
            before, used = self._last_sample
            if now > before:
                result["cpu_percent"] = max(0, (cpu_s - used) / (now - before) * 100)
        self._last_sample = (now, cpu_s)
        # A bounded dynamic PID list avoids attributing host processes to a run.
        capacity = max(16, info.ActiveProcesses + 16)
        class Pids(ctypes.Structure):
            _fields_ = [("assigned", wintypes.DWORD), ("count", wintypes.DWORD),
                        ("pids", ctypes.c_size_t * capacity)]
        pids = Pids()
        if _kernel32.QueryInformationJobObject(
            self._job, 3, ctypes.byref(pids), ctypes.sizeof(pids), None
        ):
            memory, complete = 0, True
            for pid in pids.pids[:pids.count]:
                handle = _kernel32.OpenProcess(0x1010, False, pid)
                if not handle:
                    complete = False
                    break
                try:
                    member = wintypes.BOOL()
                    if not _kernel32.IsProcessInJob(handle, self._job, ctypes.byref(member)) \
                            or not member.value:
                        complete = False
                        break
                    counters = _Memory()
                    counters.cb = ctypes.sizeof(counters)
                    if not _psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                        complete = False
                        break
                    memory += counters.PrivateUsage
                finally:
                    _kernel32.CloseHandle(handle)
            if complete:
                result["ram_mb"] = memory / 1024**2
        return result

    def kill(self) -> None:
        """Kill every process in the tree, now. Safe to call more than once."""
        if self._job is not None:
            _kernel32.TerminateJobObject(self._job, 1)
        else:
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        try:
            if hasattr(self, "process"):
                self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()

    def close(self) -> None:
        self.kill()
        if self._job is not None:
            _kernel32.CloseHandle(self._job)
            self._job = None
        if self._port is not None:
            _kernel32.CloseHandle(self._port)
            self._port = None
