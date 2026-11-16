# S020b offline audiovisual image

Habitat 1.2.0 adds FFmpeg/ffprobe to the existing S020a scientific image. The
owner's `Run next session` instruction accepted the S020b handoff and its
contained FFmpeg/reference provisioning; no host software/settings, new base
image or models are involved. Provisioning is an operator action, never a tool.

```text
wsl -d Ubuntu --exec podman build --pull=never --network=host --memory=2g --tag localhost/open-lab-workshop:s020b --file /mnt/c/open-lab-simulation/workshop/Media.Containerfile /mnt/c/open-lab-simulation/workshop
```

Substitute actual WSL workspace paths where necessary. The recipe requires the
already provisioned S020a image by immutable local configuration ID and pins
Debian FFmpeg `7:7.1.5-0+deb13u1`. Network is used for build-time package
acquisition only. Runtime remains rootless, read-only, network/GPU-free, with
workspace-only writes and the existing explicit resource/timeout/quota controls.

Local image config ID:

```text
sha256:a0fb028f8e686c80e8c317b88570fc0bd61878be2b0c267a3a14fed27e341efb
```

Size is 1,495,645,879 bytes on disk, not its RAM footprint. Exact installed
Debian package versions, FFmpeg version/build options and scientific wheel
notices are retained under `/opt/open-lab-workshop/`; Debian licences/copyright
notices remain under `/usr/share/doc/`. The Debian build enables GPL components;
do not describe it as LGPL-only. No image binary is committed or published.
Transitive Debian repository packages may change on rebuild: the resulting ID
must match the world, or a new world/image release is required. Older images and
fixtures are unchanged. Do not silently substitute the build tag for the ID.

Use python.run programs with `subprocess.run` argument lists, `-nostdin`, one
encoder/decoder/filter thread and bounded subprocess timeouts. Keep generated
files under the existing 50 MiB preservation cap. The 60 s Python maximum and
90 s outer timeout are unchanged. Small procedural media, not long films or AI
generation, is supported. Documentation and source datasets are included in
Builder's immutable `library/` release assets and exposed by library.search/read.

The committed component fixture in `tests/test_library_media.py` reads/cites
actual references, copies Iris CSV to the workspace, generates NumPy PCM audio,
Pillow PNG/GIF frames, muxes VP8/Vorbis WebM, trims/flips video, mixes audio,
extracts a PNG and inspects ffprobe/PCM/pixel results. Recorder/scanner/checkpoint
and data-only lineage components preserve identical files. No inhabitant loop,
scripted model, provider or inference is invoked.

```powershell
$env:OPEN_LAB_REQUIRE_CONTAINER='1'
uv run pytest tests/test_library_media.py tests/test_workshop.py -o addopts= -q
uv run ruff check .
```

Window independently previews the resulting WAV/WebM with native controls.
Its read-only server never invokes FFmpeg or imports Simulation code. Raw build
and optional exported test artifacts remain ignored under .local/. Existing
full-run demonstrations are historical, not this programme's verification path.
