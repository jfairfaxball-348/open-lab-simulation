# S020a scientific workshop image

Operator provisioning for Habitat 1.1.0. This image contains no models. It uses
the already approved S006c base and installs scientific Python distributions
only inside the derived image; no host packages or system settings change.
The owner's `Run 20a` instruction authorised this named workshop provisioning.

```text
wsl -d Ubuntu --exec podman build --pull=never --network=host --memory=2g --tag localhost/open-lab-workshop:s020a --file /mnt/c/open-lab-simulation/workshop/Containerfile /mnt/c/open-lab-simulation/workshop
```

The example assumes sibling checkouts at the workspace root; substitute the
actual WSL path obtained with `wslpath` when elsewhere. The build needs network
access only for approved wheel acquisition. `requirements.lock` pins all 15
distributions and their hashes; pip requires hashes and binary wheels, then
checks dependency consistency. No source build, apt install or base-image pull.
The recipe and requirements are source; image layers and raw build output are
not committed. The runtime never builds, pulls or installs packages.

Approved local image configuration ID for this release:

```text
sha256:f1a63cbdd0c4afbe35c533efa8f3ca21ff4f46c80a36d4d9cbba5bca08826f6c
```

This is a config ID, not a registry manifest digest. Podman reports its `Id`
without the algorithm prefix; Simulation normalises that exact hex form before
comparison. Launch uses the immutable ID with `--pull=never`, not the build tag.
The selected world image is inspected before allocation and is recorded in
isolation.json (and per-resume segment isolation metadata). Old worlds omit the
new field and still select the legacy S006c image. Rebuilding can produce a
different image ID: do not edit a published release or silently substitute it.
Provision/copy the exact image or issue a new world version with its new ID.

`/opt/open-lab-workshop/inventory.json` identifies all installed locked versions.
`/opt/open-lab-workshop/notices/` retains licence/notice files from the exact
wheels, including bundled native-library notices; original distribution metadata
is also retained. Root lock: NumPy 2.5.5, SciPy 1.18.1, SymPy 1.14.0, mpmath
1.3.0, pandas 3.1.1, Matplotlib 3.11.2, Pillow 13.0.0. The derived image is
595,191,083 bytes locally; this is disk size, not a RAM requirement.

Use `python.packages` to inspect actual installed versions. `python.run` retains
its 10 s default / 60 s maximum inner timeout and 90 s outer timeout; use 60 s
for science/plot imports. The runner sets one native thread, Matplotlib Agg and
workspace-only cache locations. Tests explicitly grant 2048 MiB, two CPU cores
and 64 processes; actual experiment grants remain authoritative. World defaults
and resource hints never widen an existing experiment's limits. No network,
GPU, model server, recorder or host directories are exposed to Python.

Offline verification, without any inhabitant loop or model request:

```powershell
$env:OPEN_LAB_REQUIRE_CONTAINER='1'
uv run pytest tests/test_workshop.py -o addopts= -q
uv run ruff check .
```

The committed fixture is copied byte-for-byte from Builder's generated 1.1.0
release. Never edit it or normalise embedded line endings: fixtures are `-text`
in .gitattributes because their exact bytes are digest-bound. Tests cover image
selection/refusal, legacy behaviour, symbolic/numerical work, CSV, PNG editing,
project repair/reuse, artifact revisions, sandbox limits, timeouts and quota
admission. They call ToolHost/Sandbox and recorder components directly. Existing
whole-run demonstrations remain historical and are not part of v2 verification.
