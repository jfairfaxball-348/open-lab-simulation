"""Run the inhabitant's own Python program and report what happened.

This executes agent-supplied code, so it is only ever offered by a world whose
minimum isolation tier is ``container`` and only when the experiment grants
``subprocess``: the Simulation refuses any other launch. Inside the container
there is no network, the root filesystem and the world are read-only and the
workspace is the only writable directory. The limits below (timeout, output
caps, killing the whole process group) are this capability's own; the
container's cgroup ceilings and removal are the outer boundary.
"""
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

OUTPUT_CAP = 64_000  # characters kept from each of stdout and stderr
READ_CHUNK = 8192
DEFAULT_TIMEOUT_S = 10
GRACE_S = 2


def workspace_root():
    location = os.environ.get("OPEN_LAB_WORKSPACE")
    if not location:
        raise ValueError("workspace is not configured")
    root = Path(location).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("workspace is not a directory")
    return root


def workspace_path(value):
    root = workspace_root()
    parts = value.split("/")
    if (any(c in value for c in "\\:<>\"|?*\0")
            or any(ord(c) < 32 for c in value)
            or any(p in {"", ".", ".."} or p.endswith((".", " ")) for p in parts)):
        raise ValueError("use a relative workspace path with forward slashes")
    reserved = {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"}
    reserved.update(f"{prefix}{n}" for prefix in ("COM", "LPT") for n in range(1, 10))
    current = root
    for part in parts:
        if part.split(".")[0].upper() in reserved:
            raise ValueError("reserved file name")
        current = current / part
        if current.is_symlink() or current.is_junction():
            raise ValueError("links are not supported")
    if not current.resolve().is_relative_to(root):
        raise ValueError("path leaves workspace")
    return root, current


def _environment():
    # Nothing from the capability's own environment reaches the program except
    # what an interpreter needs to start (Windows needs SYSTEMROOT).
    environment = {"PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"}
    # Native libraries otherwise oversubscribe small CPU/pid budgets; plotting
    # must work without a display or writes to a read-only home directory.
    cache = workspace_root() / ".cache"
    environment.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
                       MPLBACKEND="Agg", MPLCONFIGDIR=str(cache / "matplotlib"),
                       XDG_CACHE_HOME=str(cache))
    for name in ("SYSTEMROOT", "PATH", "TMPDIR", "TEMP", "TMP"):
        if name in os.environ:
            environment[name] = os.environ[name]
    return environment


def _collect(stream, sink):
    # Keep the first OUTPUT_CAP characters' worth of bytes, but always drain the
    # pipe so a talkative program cannot block on a full buffer.
    kept, dropped = bytearray(), 0
    limit = OUTPUT_CAP * 4
    while True:
        chunk = stream.read(READ_CHUNK)
        if not chunk:
            break
        room = limit - len(kept)
        kept.extend(chunk[:max(room, 0)])
        dropped += max(len(chunk) - max(room, 0), 0)
    sink.append((bytes(kept), dropped))


def _decode(captured):
    data, dropped = captured
    # Line endings are normalised so results read the same on every host.
    text = data.decode("utf-8", "replace").replace("\r\n", "\n")
    truncated = dropped > 0 or len(text) > OUTPUT_CAP
    return text[:OUTPUT_CAP], truncated


def _kill_group(process):
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)],  # noqa: S603, S607
                       capture_output=True, check=False)
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def run(arguments):
    has_path, has_code = "path" in arguments, "code" in arguments
    if has_path == has_code:
        raise ValueError("give either path (a workspace .py file) or code")
    root = workspace_root()
    if has_path:
        root, target = workspace_path(arguments["path"])
        if target.suffix != ".py" or not target.is_file():
            raise ValueError("path must name an existing .py file in the workspace")
        program = [str(target)]
        ran = target.relative_to(root).as_posix()
    else:
        program = ["-c", arguments["code"]]
        ran = "<code>"
    # -E/-s: ignore PYTHON* variables and the user site. Unlike -I, keep the
    # script's directory (or, for code, the workspace) on the import path, so
    # the inhabitant's own modules can import each other.
    command = [sys.executable, "-E", "-s", "-B", "-X", "utf8", *program,
               *arguments.get("args", [])]
    timeout = arguments.get("timeout_s", DEFAULT_TIMEOUT_S)
    options = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt"
               else {"start_new_session": True})
    try:
        process = subprocess.Popen(  # noqa: S603 - argument vector, never a shell
            command, cwd=root, env=_environment(), stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, **options)
    except OSError as error:
        raise ValueError(f"could not start Python: {error.strerror or error}") from None
    out, err = [], []
    readers = [threading.Thread(target=_collect, args=(process.stdout, out), daemon=True),
               threading.Thread(target=_collect, args=(process.stderr, err), daemon=True)]
    for reader in readers:
        reader.start()
    try:
        process.stdin.write(arguments.get("stdin", "").encode("utf-8"))
        process.stdin.close()
    except OSError:
        pass  # the program exited or closed its input; its exit status says why
    timed_out = False
    deadline = time.monotonic() + timeout
    while process.poll() is None:
        if time.monotonic() >= deadline:
            timed_out = True
            _kill_group(process)
            break
        time.sleep(0.02)
    try:
        process.wait(timeout=GRACE_S)
    except subprocess.TimeoutExpired:
        _kill_group(process)
        process.wait()
    # Descendants that kept the pipes open were killed with the group.
    for reader in readers:
        reader.join(GRACE_S)
    stdout, stdout_truncated = _decode(out[0] if out else (b"", 0))
    stderr, stderr_truncated = _decode(err[0] if err else (b"", 0))
    return {"ran": ran, "exit_code": None if timed_out else process.returncode,
            "timed_out": timed_out, "stdout": stdout, "stderr": stderr,
            "stdout_truncated": stdout_truncated, "stderr_truncated": stderr_truncated}
