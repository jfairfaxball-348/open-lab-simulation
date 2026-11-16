"""S020a component checks: real sandbox calls, never an inhabitant loop."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from open_lab_simulation import container
from open_lab_simulation.artifacts import WorkspaceScanner, read_index
from open_lab_simulation.recorder import Recorder
from open_lab_simulation.resources import Controller
from open_lab_simulation.supervisor import _isolation
from open_lab_simulation.toolhost import Sandbox, ToolHost
from open_lab_simulation.world import WorldError, load_world

WORLD = Path(__file__).parent / "fixtures/worlds/habitat/1.1.0"
LEGACY = Path(__file__).parent / "fixtures/worlds/habitat/1.0.0"
IMAGE = json.loads((WORLD / "world-release.json").read_text())["environment"]["container_image"]


def test_selection_and_legacy_default():
    assert container.selected_image(load_world(LEGACY).manifest) == container.BASE_IMAGE
    assert container.selected_image(load_world(WORLD).manifest) == IMAGE


@pytest.mark.parametrize("value", ["latest", "localhost/image:tag", "sha256:bad", None,
                                   "sha256:" + "A" * 64, IMAGE + "\n"])
def test_image_selector_refuses_untrusted_references(value):
    manifest = copy.deepcopy(load_world(WORLD).manifest)
    manifest["environment"]["container_image"] = value
    with pytest.raises(container.ContainerUnavailable):
        container.selected_image(manifest)


@pytest.mark.parametrize("version", ["1.0.0", "1.2.0", "2.0.0"])
def test_image_semantics_need_supported_schema(version):
    manifest = copy.deepcopy(load_world(WORLD).manifest)
    manifest["schema_version"] = version
    with pytest.raises(container.ContainerUnavailable):
        container.selected_image(manifest)


def test_supervisor_admission_selects_world_image(monkeypatch):
    observed = []

    def probe(self):
        observed.append(self.image)
        return {"tier": "container", "image": self.image, "image_id": self.image}

    monkeypatch.setattr(container.Backend, "probe", probe)
    result = _isolation({"limits": {}}, load_world(WORLD))
    assert observed == [IMAGE]
    assert result["image"] == result["image_id"] == IMAGE
    assert result["limits"] == container.DEFAULT_LIMITS


class InspectOnly(container.Backend):
    def __init__(self, actual=None, missing=False):
        super().__init__(image=IMAGE)
        self.actual, self.missing, self.calls = actual or IMAGE, missing, []

    def _control(self, *argv, **kwargs):
        self.calls.append(argv)
        if argv[:2] == ("podman", "info"):
            return json.dumps({"host": {"security": {"rootless": True},
                              "cgroupVersion": "v2", "cgroupControllers": ["cpu", "memory", "pids"],
                              "ociRuntime": {"name": "crun"}},
                               "version": {"Version": "test"}}).encode()
        assert argv == ("podman", "image", "inspect", IMAGE)
        if self.missing:
            raise container.ContainerUnavailable("image absent")
        return json.dumps([{"Id": self.actual}]).encode()


def test_probe_inspects_exact_identity_and_never_pulls(monkeypatch):
    monkeypatch.setattr(container.sys, "platform", "win32")
    backend = InspectOnly()
    assert backend.probe()["image_id"] == IMAGE
    assert InspectOnly(actual=IMAGE.removeprefix("sha256:")).probe()["image_id"] == IMAGE
    for backend, message in [(InspectOnly(actual="sha256:" + "0" * 64), "pinned digest"),
                             (InspectOnly(missing=True), "never pulled")]:
        with pytest.raises(container.ContainerUnavailable, match=message):
            backend.probe()
        assert all("pull" not in call for call in backend.calls)


def test_image_tampering_breaks_verified_world(tmp_path):
    import shutil

    target = tmp_path / "world"
    shutil.copytree(WORLD, target)
    path = target / "world-release.json"
    manifest = json.loads(path.read_text())
    manifest["environment"]["container_image"] = "sha256:" + "0" * 64
    path.write_text(json.dumps(manifest))
    with pytest.raises(WorldError, match="digest"):
        load_world(target)


@pytest.fixture
def workshop(tmp_path):
    world = load_world(WORLD)
    backend = container.Backend(image=container.selected_image(world.manifest))
    try:
        backend.probe()
    except container.ContainerUnavailable as error:
        if os.environ.get("OPEN_LAB_REQUIRE_CONTAINER") == "1":
            pytest.fail(str(error))
        pytest.skip(str(error))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    events = []
    host = ToolHost(world.capabilities, workspace,
                    {"filesystem": "workspace_write", "network": "none", "subprocess": True},
                    events.append, {"kind": "agent", "model_id": "scripted"},
                    sandbox=Sandbox(backend, WORLD.resolve(), "run-workshop-components",
                                    {"ram_mb": 2048, "cpu_cores": 2, "max_processes": 64}))
    yield host, world
    assert backend.running("run-workshop-components") == []


def invoke(host, ident, **arguments):
    reply = host.call(ident, arguments)
    assert reply["status"] == "ok", reply
    return reply["result"]


SCIENCE = '''
import io, json, os, sys
import numpy as np
from scipy.integrate import quad
import sympy as s
import mpmath as mp
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from helper import scale
assert np.mean.__doc__ and s.integrate.__doc__
x = s.symbols("x")
assert s.simplify(s.diff(s.sin(x)**2+s.cos(x)**2, x)) == 0
assert abs(quad(np.sin, 0, np.pi)[0] - 2) < 1e-10
mp.mp.dps = 40
assert str(mp.sqrt(2)).startswith("1.414213562373095")
data = pd.read_csv(io.StringIO("x,y\\n1,2\\n2,4\\n3,6\\n"))
assert data.y.mean() == 4
plt.plot(data.x, scale(data.y), color=sys.argv[1])
plt.savefig("plot.png")
with Image.open("plot.png") as image:
    assert image.width > 100 and image.height > 100
    image.crop((0,0,100,100)).save("crop.png")
assert all(os.environ[k] == "1" for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"))
assert not any(k.startswith("OPEN_LAB") for k in os.environ)
assert os.environ["MPLBACKEND"] == "Agg"
assert os.environ["MPLCONFIGDIR"].startswith("/workspace/")
open("results.json","w").write(json.dumps({"mean":float(data.y.mean()),"scaled":list(scale(data.y))}))
print("science verified")
'''


def test_real_science_project_repair_revision_and_artifact_capture(workshop, tmp_path):
    host, world = workshop
    inventory = invoke(host, "python.packages")["packages"]
    assert len(inventory) == 7 and all(p["version"] for p in inventory)
    locked = {}
    lock = Path(__file__).parent.parent / "workshop/requirements.lock"
    for line in lock.read_text().splitlines():
        if line and not line[0].isspace() and "==" in line:
            name, version = line.split()[0].split("==")
            locked[name] = version
    assert {p["name"]: p["version"] for p in inventory} == {
        p["name"]: locked[p["name"]] for p in inventory}
    invoke(host, "workspace.write", path="helper.py",
           text="def scale(x):\n    return missing * x\n")
    invoke(host, "workspace.write", path="study.py", text=SCIENCE)
    broken = invoke(host, "python.run", path="study.py", args=["blue"], timeout_s=60)
    assert broken["exit_code"] == 1 and "NameError" in broken["stderr"]
    invoke(host, "workspace.edit", path="helper.py", old_text="missing", new_text="2")
    good = invoke(host, "python.run", path="study.py", args=["blue"], timeout_s=60)
    assert good["exit_code"] == 0, good
    assert good["stdout"] == "science verified\n"
    assert json.loads(invoke(host, "workspace.read", path="results.json")["text"]) == {
        "mean": 4, "scaled": [4, 8, 12]}
    archive = tmp_path / "archive"
    archive.mkdir()
    recorder = Recorder(archive / "events.jsonl", world.contracts, "workshop-check",
                        "run-workshop-components")
    try:
        scanner = WorkspaceScanner(host.workspace, archive, recorder, world.contracts,
                                   {"kind": "supervisor"})
        scanner.scan({"kind": "tool", "capability_id": "python.run"})
        blue = (host.workspace / "plot.png").read_bytes()
        assert blue.startswith(b"\x89PNG\r\n\x1a\n")
        good = invoke(host, "python.run", path="study.py", args=["red"], timeout_s=60)
        assert good["exit_code"] == 0, good
        red = (host.workspace / "plot.png").read_bytes()
        assert blue != red
        scanner.scan({"kind": "tool", "capability_id": "python.run"})
        versions = [r for r in read_index(archive) if r["workspace_path"] == "plot.png"]
        assert [r["revision"] for r in versions] == [1, 2]
        for record, data in zip(versions, [blue, red], strict=True):
            assert record["sha256"] == hashlib.sha256(data).hexdigest()
            assert (archive / record["archive_path"]).read_bytes() == data
    finally:
        recorder.close()


def test_workshop_isolation_timeout_and_quota(workshop, tmp_path):
    host, world = workshop
    result = invoke(host, "python.run", code='''
import json, os, pathlib, socket
blocked = []
for path in ("/world/escape", "/usr/escape", "/escape"):
    try: pathlib.Path(path).write_text("x")
    except OSError: blocked.append(path)
assert len(blocked) == 3 and not os.path.exists("/mnt/c")
for target in (("1.1.1.1",53),("127.0.0.1",11434)):
    try: socket.create_connection(target,1)
    except OSError: pass
    else: raise AssertionError("network available")
assert open("/sys/fs/cgroup/memory.max").read().strip() == str(2048*1024**2)
print("isolated")
''')
    assert result["exit_code"] == 0, result
    timeout = invoke(host, "python.run", code="import time; time.sleep(120)", timeout_s=1)
    assert timeout["timed_out"] and timeout["exit_code"] is None
    invoke(host, "workspace.write", path="quota.bin", text="x" * 1000)
    recorder = SimpleNamespace(record=lambda **kwargs: None)
    controller = Controller({"limits": {"resources": {"disk_mb": 1}}}, host.workspace,
                            world, recorder)
    (host.workspace / "full.bin").write_bytes(b"x" * (1024**2))
    host.admission = lambda: controller.admit("python.run")
    denied = host.call("python.run", {"code": "open('forbidden','w').write('x')"})
    assert denied["status"] == "denied" and "quota" in denied["error"]
    assert not (host.workspace / "forbidden").exists()
