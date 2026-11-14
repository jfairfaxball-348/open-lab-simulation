"""The container isolation tier (S006c): launch gating, and escape/limit fixtures.

The fixtures below run on the actual backend: rootless Podman in WSL 2. When
the backend is absent they are skipped with the probe's reason, unless
``OPEN_LAB_REQUIRE_CONTAINER=1`` is set, in which case they fail. Milestone
evidence is only ever a run with that variable set.

``test.escape`` executes the code it is given. It stands in for agent-supplied
code; the snippets it runs here are fixed test fixtures.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import time
from pathlib import Path

import pytest

from conftest import (
    FIXTURE_WORLD,
    TEST_CAPABILITIES,
    add_test_capabilities,
    call,
    events_of,
    restamp,
)
from open_lab_simulation import container
from open_lab_simulation.archive import verify_archive
from open_lab_simulation.checkpoints import CheckpointError, _workspace_snapshot
from open_lab_simulation.paths import is_link, walk_files
from open_lab_simulation.resources import workspace_bytes
from open_lab_simulation.supervisor import LaunchError, _unsupported, run_experiment
from open_lab_simulation.toolhost import Sandbox, ToolHost
from open_lab_simulation.world import load_world

REPO = Path(__file__).resolve().parent.parent
REQUIRE = os.environ.get("OPEN_LAB_REQUIRE_CONTAINER") == "1"

ESCAPE = '''
def run(arguments):
    scope = {}
    exec(arguments["code"], scope)
    return {"ok": True, "result": scope.get("result")}
'''
SLEEP = '''
import time
def run(arguments):
    time.sleep(arguments.get("seconds", 300))
    return {"ok": True}
'''
GROW = '''
import os, time
from pathlib import Path
def run(arguments):
    with (Path(os.environ["OPEN_LAB_WORKSPACE"]) / "growing.txt").open("wb") as f:
        for _ in range(1000):
            f.write(b"x" * 65536)
            f.flush()
            time.sleep(0.01)
    return {"ok": True}
'''
LIMITS = {"ram_mb": 256, "cpu_cores": 0.5, "max_processes": 16}


# --- launch gating and command construction: no backend needed ---------------------


class FakeBackend(container.Backend):
    def host_path(self, path: Path) -> str:
        return "/mnt/x/" + path.as_posix().replace(":", "")


def vector(**changes):
    world = Path("C:/data/worlds/w/1.0.0")
    arguments = dict(name="ol-run-1-ab", run_id="run-1", world=world,
                     capability_dir=world / "capabilities" / "a.b", workspace=Path("C:/ws"),
                     writable=True, argv=["python", "-I"], env={"K": "V"},
                     limits=container.container_limits(None), timeout_s=5)
    arguments.update(changes)
    return FakeBackend().command(**arguments)


def test_command_denies_everything_not_granted():
    argv = vector()
    assert argv[:5] == ["wsl.exe", "-d", container.DISTRO, "--exec", "podman"]
    for flag in ("--network=none", "--read-only", "--cap-drop=ALL", "--pull=never",
                 "--security-opt=no-new-privileges", "--userns=keep-id", "--rm",
                 "--memory=512m", "--memory-swap=512m", "--cpus=1.0", "--pids-limit=64",
                 "--timeout=5"):
        assert flag in argv, flag
    assert argv[argv.index("--label") + 1] == f"{container.RUN_LABEL}=run-1"
    assert "--env-host" not in argv and "--privileged" not in argv
    volumes = [argv[i + 1] for i, item in enumerate(argv) if item == "--volume"]
    assert volumes == ["/mnt/x/C/data/worlds/w/1.0.0:/world:ro,nosuid,nodev,noexec",
                       "/mnt/x/C/ws:/workspace:rw,nosuid,nodev,noexec"]
    assert argv[argv.index("--workdir") + 1] == "/world/capabilities/a.b"
    assert argv[-3:] == [container.BASE_IMAGE, "python", "-I"]
    assert "@sha256:" in container.BASE_IMAGE


def test_workspace_mount_follows_the_filesystem_permission():
    read_only = vector(writable=False)
    assert "/mnt/x/C/ws:/workspace:ro,nosuid,nodev,noexec" in read_only
    assert not any(item.endswith(":/workspace:ro,nosuid,nodev,noexec") or
                   "/workspace:" in item for item in vector(workspace=None))


def test_container_limits_default_and_refuse_unusable_values():
    assert container.container_limits(None) == container.DEFAULT_LIMITS
    assert container.container_limits({"ram_mb": 300, "disk_mb": 5}) == {
        **container.DEFAULT_LIMITS, "ram_mb": 300}
    with pytest.raises(container.ContainerUnavailable):
        container.container_limits({"ram_mb": container.TMPFS_MB})


def container_world(release: Path, *, network_tool: bool = False) -> None:
    def change(manifest):
        manifest["environment"]["isolation"]["minimum_tier"] = "container"
    restamp(release, change)
    if network_tool:
        path = release / "capabilities" / "time.now" / "capability.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["permissions"]["network"] = "loopback"
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
        from open_lab_simulation.world import file_sha256

        def fix(document):
            for entry in document["capabilities"]:
                if entry["id"] == "time.now":
                    entry["sha256"] = file_sha256(path)
        restamp(release, fix)


def test_container_world_refused_when_backend_probe_fails(root, release, make_config,
                                                          monkeypatch):
    container_world(release)

    def unavailable(self):
        raise container.ContainerUnavailable("probe says no")
    monkeypatch.setattr(container.Backend, "probe", unavailable)
    with pytest.raises(LaunchError, match="container isolation tier is not available: probe"):
        run_experiment(make_config([]), root)
    assert not (root / "runs").exists()


def test_container_world_refuses_capabilities_needing_network(release):
    container_world(release, network_tool=True)
    world = load_world(release)
    config = {"termination": {"on_deadline": "hard_stop"}, "limits": {},
              "permissions": {"filesystem": "workspace_write", "network": "none"}}
    assert "time.now" in _unsupported(config, world)


def test_subprocess_permission_needs_a_container_world(release):
    config = {"termination": {"on_deadline": "hard_stop"}, "limits": {},
              "permissions": {"filesystem": "workspace_write", "network": "none",
                              "subprocess": True}}
    assert "container isolation tier" in _unsupported(config, load_world(release))
    container_world(release)
    assert _unsupported(config, load_world(release)) is None


# --- the actual backend ---------------------------------------------------------------


@pytest.fixture(scope="session")
def backend():
    found = container.Backend()
    try:
        found.probe()
    except container.ContainerUnavailable as error:
        if REQUIRE:
            pytest.fail(f"container backend required but unavailable: {error}")
        pytest.skip(f"container backend unavailable: {error}")
    return found


def manifest_for(ident, permissions, timeout_s):
    template = json.loads((FIXTURE_WORLD / "capabilities/text.stats/capability.json")
                          .read_text(encoding="utf-8"))
    manifest = copy.deepcopy(template)
    manifest.update(id=ident, name=ident,
                    invocation={"interface": "python", "entrypoint": "impl:run"},
                    input_schema={"type": "object"},
                    output_schema={"type": "object", "required": ["ok"]},
                    permissions=permissions, resources={"timeout_s": timeout_s})
    return manifest


@pytest.fixture
def sandboxed(backend, tmp_path):
    """A tool host whose capabilities run in containers, with escape fixtures."""
    world = tmp_path / "world"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    tools = {
        "test.escape": ("workspace_write", 60, ESCAPE),
        "test.escape-read": ("workspace_read", 60, ESCAPE),
        "test.escape-none": ("none", 60, ESCAPE),
        "test.sleep": ("none", 3, SLEEP),
    }
    capabilities = {}
    for ident, (filesystem, timeout_s, code) in tools.items():
        directory = world / "capabilities" / ident
        directory.mkdir(parents=True)
        (directory / "impl.py").write_text(code, encoding="utf-8", newline="\n")
        capabilities[ident] = (directory, manifest_for(
            ident, {"filesystem": filesystem, "network": "none"}, timeout_s))
    run_id = f"run-test-{os.urandom(3).hex()}"
    events = []
    host = ToolHost(capabilities, workspace,
                    {"filesystem": "workspace_write", "network": "none"}, events.append,
                    {"kind": "agent", "model_id": "scripted"},
                    sandbox=Sandbox(backend, world, run_id, dict(LIMITS)))
    host.run_id = run_id
    yield host
    backend.sweep(run_id)


def escape(host, code, ident="test.escape"):
    outcome = host.call(ident, {"code": code})
    assert outcome["status"] == "ok", outcome
    return outcome["result"]["result"]


def test_host_files_repositories_and_archives_are_unreachable(sandboxed, backend, tmp_path):
    archive = tmp_path / "runs" / "e" / "run-x"
    archive.mkdir(parents=True)
    hidden = [backend.host_path(path) for path in
              (REPO, REPO.parent, archive, tmp_path, Path.home())]
    found = escape(sandboxed, f'''
import os
paths = {hidden!r} + ["/mnt", "/mnt/c", "/mnt/wsl", "/run/WSL", "/init", "/home", "/root/.ssh"]
mounts = [line.split()[3:5] + [line.split(" - ")[1].split()[0]]
          for line in open("/proc/self/mountinfo")]
result = {{"exists": [p for p in paths if os.path.exists(p)],
          "host_mounts": [m for m in mounts if m[2] in ("9p", "drvfs", "virtiofs")],
          "mountpoints": sorted(m[1] for m in mounts)}}
''')
    # The base image ships /mnt and /home as empty directories; nothing else may exist.
    assert set(found["exists"]) <= {"/mnt", "/home"}
    assert sorted(m[1] for m in found["host_mounts"]) == ["/workspace", "/world"]
    assert all(not point.startswith("/mnt/") for point in found["mountpoints"])
    assert escape(sandboxed, 'import os\nresult = os.listdir("/mnt") + os.listdir("/home")') == []


def test_network_is_absent(sandboxed):
    found = escape(sandboxed, '''
import os, socket
attempts = {}
for host, port in (("1.1.1.1", 53), ("8.8.8.8", 443), ("127.0.0.1", 11434),
                   ("10.255.255.254", 53), ("172.17.0.1", 80)):
    try:
        socket.create_connection((host, port), 2).close()
        attempts[f"{host}:{port}"] = "connected"
    except OSError as error:
        attempts[f"{host}:{port}"] = error.__class__.__name__
try:
    socket.getaddrinfo("example.com", 443)
    attempts["dns"] = "resolved"
except OSError as error:
    attempts["dns"] = error.__class__.__name__
result = {"interfaces": sorted(os.listdir("/sys/class/net")), "attempts": attempts}
''')
    assert found["interfaces"] == ["lo"]
    assert "connected" not in found["attempts"].values()
    assert found["attempts"]["dns"] != "resolved"


def test_no_privileges_and_no_writes_outside_the_workspace(sandboxed):
    found = escape(sandboxed, '''
import ctypes, os
status = dict(line.split(":\\t", 1) for line in open("/proc/self/status").read().splitlines()
              if ":\\t" in line)
writes = {}
for path in ("/etc/x", "/usr/x", "/x", "/world/x", "/world/capabilities/test.escape/impl.py",
             "/workspace/ok.txt", "/tmp/ok.txt"):
    try:
        with open(path, "a") as handle:
            handle.write("x")
        writes[path] = "written"
    except OSError as error:
        writes[path] = error.__class__.__name__
try:
    os.setuid(0)
    setuid = "became root"
except OSError as error:
    setuid = error.__class__.__name__
libc = ctypes.CDLL(None, use_errno=True)
mounted = libc.mount(b"none", b"/tmp", b"tmpfs", 0, None)
result = {"uid": os.getuid(), "caps": [status[k].strip() for k in ("CapEff", "CapPrm", "CapBnd")],
          "no_new_privs": status["NoNewPrivs"].strip(), "writes": writes, "setuid": setuid,
          "mount": [mounted, ctypes.get_errno()]}
''')
    assert found["uid"] != 0
    assert set(found["caps"]) == {"0000000000000000"}
    assert found["no_new_privs"] == "1"
    assert found["setuid"] == "PermissionError"
    assert found["mount"][0] == -1
    written = {path for path, outcome in found["writes"].items() if outcome == "written"}
    assert written == {"/workspace/ok.txt", "/tmp/ok.txt"}
    assert (sandboxed.workspace / "ok.txt").read_text() == "x"


def test_workspace_permission_is_enforced_by_the_mount(sandboxed):
    (sandboxed.workspace / "seen.txt").write_text("seen", encoding="utf-8")
    code = '''
import os
outcome = {"visible": os.path.exists("/workspace"), "env": os.environ.get("OPEN_LAB_WORKSPACE")}
try:
    outcome["read"] = open("/workspace/seen.txt").read()
    open("/workspace/new.txt", "w").write("x")
    outcome["write"] = "written"
except OSError as error:
    outcome["write"] = error.__class__.__name__
result = outcome
'''
    read_only = escape(sandboxed, code, "test.escape-read")
    assert (read_only["visible"], read_only["env"], read_only["read"]) == (
        True, "/workspace", "seen")
    assert read_only["write"] != "written"
    assert not (sandboxed.workspace / "new.txt").exists()
    nothing = escape(sandboxed, code, "test.escape-none")
    assert nothing["visible"] is False and nothing["env"] is None


def test_written_files_cannot_be_executed(sandboxed):
    found = escape(sandboxed, '''
import os, subprocess
outcome = {}
for directory in ("/workspace", "/tmp"):
    for name, content in (("script", b"#!/bin/sh\\necho escaped\\n"),
                          ("windows.exe", b"MZ" + b"\\0" * 256)):
        path = f"{directory}/{name}"
        with open(path, "wb") as handle:
            handle.write(content)
        os.chmod(path, 0o755)
        try:
            outcome[path] = subprocess.run([path], capture_output=True, timeout=10).returncode
        except OSError as error:
            outcome[path] = error.__class__.__name__
result = outcome
''')
    assert set(found.values()) == {"PermissionError"}, found


def test_environment_carries_nothing_from_the_host(sandboxed):
    names = escape(sandboxed, 'import os\nresult = sorted(os.environ)')
    allowed = {"PATH", "LANG", "HOME", "HOSTNAME", "TERM", "container", "GPG_KEY",
               "PYTHON_VERSION", "PYTHON_SHA256", "OPEN_LAB_WORKSPACE",
               "PYTHONDONTWRITEBYTECODE", "PYTHONIOENCODING"}
    assert set(names) <= allowed, set(names) - allowed


def test_cgroup_ceilings_are_the_configured_ones(sandboxed):
    found = escape(sandboxed, '''
result = [open(f"/sys/fs/cgroup/{name}").read().split() for name in
          ("memory.max", "memory.swap.max", "pids.max", "cpu.max")]
''')
    assert found == [[str(256 * 1024**2)], ["0"], ["16"], ["50000", "100000"]]


def test_process_count_is_bounded(sandboxed):
    started = escape(sandboxed, '''
import subprocess
children = []
try:
    for _ in range(100):
        children.append(subprocess.Popen(["sleep", "30"]))
except OSError:
    pass
result = len(children)
for child in children:
    child.kill()
''')
    assert started < LIMITS["max_processes"]


def test_memory_ceiling_kills_the_container(sandboxed):
    outcome = sandboxed.call("test.escape", {"code": '''
blocks = []
while True:
    blocks.append(bytearray(16 * 1024 * 1024))
'''})
    assert outcome["status"] == "error"
    assert "exit status 137" in outcome["error"]
    assert sandboxed.sandbox.backend.running(sandboxed.run_id) == []


def test_cpu_ceiling_throttles(sandboxed):
    used = escape(sandboxed, '''
import os, time
def usage():
    return int(dict(l.split() for l in open("/sys/fs/cgroup/cpu.stat"))["usage_usec"])
before, start = usage(), time.monotonic()
children = []
for _ in range(2):
    pid = os.fork()
    if pid == 0:
        end = time.monotonic() + 3
        while time.monotonic() < end:
            pass
        os._exit(0)
    children.append(pid)
for pid in children:
    os.waitpid(pid, 0)
result = [usage() - before, time.monotonic() - start]
''')
    cpu_seconds, wall = used[0] / 1e6, used[1]
    # Two busy processes for three seconds would use six CPU seconds unthrottled.
    assert cpu_seconds <= LIMITS["cpu_cores"] * wall * 1.2 + 0.2, used


def test_in_memory_tmp_is_bounded(sandboxed):
    written = escape(sandboxed, '''
written = 0
try:
    with open("/tmp/fill", "wb") as handle:
        for _ in range(200):
            handle.write(b"x" * 1024 * 1024)
            handle.flush()
            written += 1
except OSError as error:
    written = [written, error.errno]
result = written
''')
    assert isinstance(written, list) and written[1] == 28  # ENOSPC
    assert written[0] <= container.TMPFS_MB


def test_timeout_removes_the_container(sandboxed):
    started = time.monotonic()
    outcome = sandboxed.call("test.sleep", {"seconds": 120})
    assert outcome["status"] == "timeout"
    assert time.monotonic() - started < 3 + container.STARTUP_ALLOWANCE_S
    assert sandboxed.sandbox.backend.running(sandboxed.run_id) == []


def test_descendants_end_with_the_call(sandboxed):
    escape(sandboxed, '''
import os, time
if os.fork() == 0:
    os.setsid()
    time.sleep(300)
    os._exit(0)
result = "returned"
''')
    assert sandboxed.sandbox.backend.running(sandboxed.run_id) == []


def test_links_made_inside_are_not_followed_by_the_host(sandboxed):
    escape(sandboxed, '''
import os
os.symlink("/mnt/c/Windows/win.ini", "/workspace/absolute")
os.symlink("../../..", "/workspace/relative")
os.symlink("kept.txt", "/workspace/to-file")
open("/workspace/kept.txt", "w").write("kept")
result = sorted(os.listdir("/workspace"))
''')
    workspace = sandboxed.workspace
    links = [workspace / name for name in ("absolute", "relative", "to-file")]
    # Windows sees WSL reparse points that Path.is_symlink() does not recognise.
    assert all(is_link(path) for path in links)
    assert not is_link(workspace / "kept.txt")
    assert list(walk_files(workspace)) == [workspace / "kept.txt"]
    with pytest.raises(ValueError, match="cannot follow links"):
        workspace_bytes(workspace)
    with pytest.raises(CheckpointError, match="links"):
        _workspace_snapshot(workspace, sandboxed.workspace.parent / "snapshot.zip", lambda: None)
    # A WSL directory link is removed as a directory entry; its target is untouched.
    for path in links:
        (os.rmdir if os.lstat(path).st_file_attributes & 0x10 else os.remove)(path)


# --- whole runs through the supervisor -----------------------------------------------


@pytest.fixture
def container_root(backend, monkeypatch, tmp_path):
    for name, spec in {
        "test.escape": ({"filesystem": "workspace_write", "network": "none"}, 60, ESCAPE),
        "test.sleep": ({"filesystem": "none", "network": "none"}, 120, SLEEP),
        "test.grow": ({"filesystem": "workspace_write", "network": "none"}, 30, GROW),
    }.items():
        monkeypatch.setitem(TEST_CAPABILITIES, name, spec)
    root = tmp_path / "data"
    release = root / "worlds/minimal/0.6.0"
    shutil.copytree(FIXTURE_WORLD, release)
    add_test_capabilities(release)
    container_world(release)
    return root


def supervisor_actions(result):
    return [e["payload"] for e in events_of(result.archive) if e["type"] == "supervisor.action"]


def test_container_run_completes_records_isolation_and_leaves_nothing(container_root,
                                                                      make_config, backend):
    result = run_experiment(make_config([
        {"tool_calls": [call("text.stats", text="one two")]},
        {"tool_calls": [call("math.evaluate", expression="2+3")]},
        {"tool_calls": [call("memory.write", key="k", value="remembered")]},
        {"tool_calls": [call("memory.search", query="remembered")]},
        {"tool_calls": [call("workspace.write", path="notes.txt", text="hello")]},
        {"tool_calls": [call("test.escape", code="import os\nresult = os.getcwd()")]},
    ], wall_clock_s=300), container_root)
    assert result.state == "completed", result.reason
    results = [e["payload"] for e in events_of(result.archive) if e["type"] == "tool.result"]
    assert [r["status"] for r in results] == ["ok"] * 6, [r.get("error") for r in results]
    assert results[1]["result"]["value"] == 5
    assert results[5]["result"]["result"] == "/world/capabilities/test.escape"
    assert (result.workspace / "notes.txt").read_text(encoding="utf-8") == "hello"
    isolation = json.loads((result.archive / "isolation.json").read_text(encoding="utf-8"))
    assert isolation["tier"] == "container" and isolation["rootless"] is True
    assert isolation["image"] == container.BASE_IMAGE
    assert isolation["limits"] == container.DEFAULT_LIMITS
    summary = json.loads((result.archive / "summary.json").read_text(encoding="utf-8"))
    assert summary["isolation"] == {"tier": "container", "image": container.BASE_IMAGE}
    assert verify_archive(result.archive) == []
    assert backend.running(result.run_id) == []


def test_deadline_stops_a_running_container(container_root, make_config, backend):
    result = run_experiment(make_config([
        {"tool_calls": [call("test.sleep", seconds=100)]},
    ], wall_clock_s=12), container_root)
    assert (result.state, result.reason) == ("terminated", "deadline")
    assert any(a.get("extensions", {}).get("stage") == "container_cleanup"
               for a in supervisor_actions(result))
    assert backend.running(result.run_id) == []
    assert verify_archive(result.archive) == []


def test_quota_stops_a_containerised_writer(container_root, make_config, backend):
    result = run_experiment(make_config([
        {"tool_calls": [call("test.grow")]},
    ], limits={"wall_clock_s": 120, "resources": {"disk_mb": 1}}), container_root)
    assert (result.state, result.reason) == ("terminated", "resource_limit:disk_mb")
    assert backend.running(result.run_id) == []
    path = result.workspace / "growing.txt"
    size = path.stat().st_size
    time.sleep(1)
    assert path.stat().st_size == size
    assert verify_archive(result.archive) == []
