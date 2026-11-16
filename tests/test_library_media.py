"""S020b real offline media and continuation components; no inhabitant loop."""

import json
import os
from pathlib import Path

import pytest

from open_lab_simulation import container
from open_lab_simulation.artifacts import WorkspaceScanner, read_index
from open_lab_simulation.checkpoints import read_checkpoint, restore_workspace, write_checkpoint
from open_lab_simulation.lineage import Source, verify
from open_lab_simulation.recorder import Recorder
from open_lab_simulation.toolhost import Sandbox, ToolHost
from open_lab_simulation.world import load_world
from test_workshop import invoke

WORLD = Path(__file__).parent / "fixtures/worlds/habitat/1.2.0"


@pytest.fixture
def media_workshop(tmp_path):
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
    host = ToolHost(world.capabilities, workspace,
                    {"filesystem": "workspace_write", "network": "none", "subprocess": True},
                    lambda event: None, {"kind": "agent", "model_id": "scripted"},
                    sandbox=Sandbox(backend, WORLD.resolve(), "run-media-components",
                                    {"ram_mb": 2048, "cpu_cores": 2, "max_processes": 64}))
    yield host, world
    assert backend.running("run-media-components") == []


RENDER = r'''
import json, subprocess, wave
from pathlib import Path
import numpy as np
import pandas as pd
import sympy as sy
import mpmath as mp
from scipy.integrate import quad
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
data = pd.read_csv("iris.csv")
assert len(data) == 150
x = sy.symbols("x")
assert sy.simplify(sy.sin(x)**2 + sy.cos(x)**2) == 1
assert abs(quad(np.sin,0,np.pi)[0] - 2) < 1e-10
assert abs(float(mp.sqrt(2))**2 - 2) < 1e-12
plt.plot(data.sepal_length_cm, data.petal_length_cm, ".")
plt.savefig("research.png")
plt.close()
frames = Path("frames")
frames.mkdir(exist_ok=True)
images = []
for i in range(12):
    image = Image.new("RGB", (128,96), (20,30,60))
    ImageDraw.Draw(image).rectangle((i*8,32,i*8+20,60),fill=(240,100,30))
    image.save(frames / f"{i:03d}.png")
    images.append(image)
images[0].save("animation.gif",save_all=True,append_images=images[1:],duration=167,loop=0)
rate = 22050
t = np.arange(rate*2)/rate
signal = (0.3*np.sin(2*np.pi*440*t)+0.15*np.sin(2*np.pi*660*t))*np.minimum(t*20,1)
with wave.open("melody.wav","wb") as audio:
    audio.setparams((1,2,rate,0,"NONE","not compressed"))
    audio.writeframes((signal*16000).astype("<i2").tobytes())
def ff(*args):
    subprocess.run(["ffmpeg","-nostdin","-hide_banner","-loglevel","error",
                    "-threads","1","-filter_threads","1","-y",*args],
                   check=True,capture_output=True,timeout=20)
ff("-framerate","6","-i","frames/%03d.png","-i","melody.wav",
   "-c:v","libvpx","-threads","1","-deadline","realtime","-cpu-used","8",
   "-b:v","150k","-c:a","libvorbis","-shortest","film.webm")
ff("-ss","0.3","-i","film.webm","-t","1","-vf","hflip",
   "-c:v","libvpx","-threads","1","-deadline","realtime","-c:a","libvorbis","clip.webm")
ff("-i","clip.webm","-frames:v","1","extracted.png")
ff("-i","melody.wav","-i","melody.wav","-filter_complex",
   "amix=inputs=2:normalize=0,volume=0.5","-c:a","pcm_s16le","mixed.wav")
probe = subprocess.run(["ffprobe","-v","error","-show_streams","-show_format",
                        "-of","json","film.webm"],check=True,capture_output=True,
                       text=True,timeout=10)
metadata = json.loads(probe.stdout)
video = next(s for s in metadata["streams"] if s["codec_type"] == "video")
assert (video["width"],video["height"]) == (128,96)
assert video["codec_name"] == "vp8"
assert any(s["codec_type"]=="audio" for s in metadata["streams"])
assert 1.9 < float(metadata["format"]["duration"]) < 2.5
with Image.open("extracted.png") as extracted:
    assert extracted.size == (128,96)
with wave.open("mixed.wav") as mixed:
    assert mixed.getnframes() == rate*2 and mixed.getnchannels() == 1
assert all(Path(f).stat().st_size < 50*1024**2 for f in
           ("film.webm","clip.webm","melody.wav","animation.gif"))
Path("analysis.json").write_text(json.dumps({"rows":len(data),"duration":metadata["format"]["duration"]}))
print("offline audiovisual work verified")
'''


def test_references_media_artifacts_checkpoint_and_lineage(media_workshop, tmp_path):
    host, world = media_workshop
    found = invoke(host, "library.search", query="integration calculus", limit=10)
    assert {s["source"]["id"] for s in found["matches"]} >= {
        "sympy-calculus", "scipy-integration"}
    citations = []
    for ident in ("sympy-calculus", "scipy-integration"):
        result = invoke(host, "library.read", source_id=ident, line_count=40)
        assert result["text"] and result["source"]["citation"]
        citations.append(f'{ident} lines 1-40: {result["source"]["citation"]}')
    invoke(host, "workspace.write", path="references.md", text="\n".join(citations))
    dataset = invoke(host, "library.read", source_id="iris", line_count=200)
    invoke(host, "workspace.write", path="iris.csv", text=dataset["text"])
    invoke(host, "workspace.write", path="render.py", text=RENDER)
    result = invoke(host, "python.run", path="render.py", timeout_s=60)
    assert result["exit_code"] == 0, result
    assert result["stdout"] == "offline audiovisual work verified\n"
    # Protected component evidence, not an allocated run or model conversation.
    archive = tmp_path / "sources" / "run-media-components"
    archive.mkdir(parents=True)
    recorder = Recorder(archive / "events.jsonl", world.contracts, "sources",
                        "run-media-components")
    try:
        scanner = WorkspaceScanner(host.workspace, archive, recorder, world.contracts,
                                   {"kind": "supervisor"})
        scanner.scan({"kind": "tool", "capability_id": "python.run"})
        preserved = {r["workspace_path"]: r for r in read_index(archive)}
        for name in ("film.webm", "clip.webm", "melody.wav", "mixed.wav", "animation.gif",
                     "research.png"):
            assert preserved[name]["preserved"]
            assert (archive / preserved[name]["archive_path"]).read_bytes() == (
                host.workspace / name).read_bytes()
        state = {"messages": [{"role": "system", "content": "component fixture"},
                              {"role": "user", "content": "no model is invoked"}],
                 "steps": 0, "script_cursor": 0, "call_count": 0, "tokens_used": 0}
        write_checkpoint(archive, host.workspace, recorder, world, "scripted", state, 0,
                         "interval")
        events = [json.loads(line) for line in (archive / "events.jsonl").read_text().splitlines()]
        metadata, restored = read_checkpoint(archive, events[-1], events, world.contracts)
        resume = tmp_path / "restored"
        restore_workspace(archive, metadata, restored, resume)
        child = tmp_path / "children" / "run-media-child"
        child.mkdir(parents=True)
        child_workspace = tmp_path / "child-workspace"
        pin = Source(archive, metadata, restored).install(child, child_workspace)
        for path in ("film.webm", "melody.wav", "animation.gif", "render.py", "references.md",
                     "research.png"):
            assert (host.workspace / path).read_bytes() == (resume / path).read_bytes()
            assert (host.workspace / path).read_bytes() == (child_workspace / path).read_bytes()
        child_events = [{"experiment_id": "children", "type": "run.state_changed",
                         "payload": {"to": "created", "extensions": {"lineage_sha256": pin}}}]
        assert verify(child, child_events) == []
        # Optional ignored export for cross-repository/browser validation only.
        if os.environ.get("OPEN_LAB_MEDIA_EXPORT"):
            export = Path(os.environ["OPEN_LAB_MEDIA_EXPORT"])
            export.mkdir(parents=True, exist_ok=True)
            for name in ("film.webm", "melody.wav"):
                (export / name).write_bytes((host.workspace / name).read_bytes())
    finally:
        recorder.close()


def test_library_cannot_be_overwritten_or_used_for_path_access(media_workshop):
    host, _ = media_workshop
    assert host.call("library.read", {"source_id": "../escape"})["status"] != "ok"
    result = invoke(host, "python.run", code='''
from pathlib import Path
import socket, subprocess
try: Path("/world/library/iris.csv").write_text("x")
except OSError: print("read-only")
else: raise AssertionError("library is writable")
assert not Path("/mnt/c").exists() and not Path("/dev/nvidia0").exists()
for endpoint in (("1.1.1.1",53),("127.0.0.1",11434)):
    try: socket.create_connection(endpoint,1)
    except OSError: pass
    else: raise AssertionError("network available")
Path("malformed.webm").write_bytes(b"not a media file")
result = subprocess.run(["ffprobe","-v","error","-protocol_whitelist","file,pipe",
                         "malformed.webm"],capture_output=True,timeout=5)
assert result.returncode != 0
''')
    assert result["exit_code"] == 0 and result["stdout"] == "read-only\n"
