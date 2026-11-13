# The Open Laboratory — Simulation

Runs one experiment: a single initial prompt, an isolated workspace, an agent
loop under a supervisor, and a record of everything observable that happened.

This is one of three independent repositories. The **World Builder**
(`open-lab-builder`) engineers the world and owns the architecture, contracts
and roadmap. The **Window** (`open-lab-window`) is a read-only view of what
this repository records. They share contract schemas and a data directory,
never code.

## Status

**S003a (runtime skeleton) is complete.** What exists and works:

- `ols run <experiment.json>` — verifies the world release, creates a fresh
  workspace and run archive, starts the agent runtime as a child process and
  supervises it to the end.
- A supervisor that enforces the wall-clock deadline and the step limit, kills
  the whole process tree, and decides how the run ends.
- A single recorder, in the supervisor, writing a gap-free, hash-chained
  `events.jsonl`. The agent runtime has no access to the archive.
- A tool host that runs the world's capabilities, checks permissions and
  arguments first, and records every call and result.
- Workspace scanning that records created, modified and deleted files and
  preserves every version in the archive.
- `ols verify <archive>` — checks an archive is complete and untampered using
  only what is inside it.

What does **not** exist yet:

- **No model is called.** The only model client is a scripted stand-in that
  replays fixed replies from the experiment configuration. Its events carry
  the model id `scripted`. A scripted run is an engineering check and says
  nothing about how any agent behaves.
- No capability discovery, memory, checkpoints, crash recovery, resource
  limits beyond the deadline and step count, or container isolation.
  Configurations that ask for any of these are refused, not approximated.
- Isolation is the *process tier*: an architectural boundary for capabilities
  written by the Builder, not a sandbox. Nothing here is safe for running
  agent-written code.
- Process-tree termination has only been run on Windows.

## Quick start

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/), and a world release
in the data root (from the Builder: `uv run olb world build worlds/minimal --publish`).

```bash
uv sync
```

```bash
uv run pytest
```

```bash
uv run ols run experiments/scripted-smoke.json
```

```bash
uv run ols verify <archive directory printed by the run>
```

The data root is `$OPEN_LAB_DATA`, or `open-lab-data` in the home directory:

```
worlds/<world_id>/<version>/        world releases (read, never written)
runs/<experiment_id>/<run_id>/      run archives (written only by the supervisor)
workspaces/<run_id>/                the only place a run's capabilities may write
```

## How a run works

```
ols run (supervisor process)
  ├─ verifies the world release against its content digest
  ├─ validates the experiment against the contracts bundled in that release
  ├─ creates the archive and an empty workspace; owns the recorder
  └─ agent runtime (child process; knows the workspace, not the archive)
       ├─ model client  → scripted stand-in
       └─ tool host     → one child process per capability call
```

The runtime asks the supervisor, over a pipe, to record what it did, and waits
for the acknowledgement before continuing. The supervisor accepts only the
event types that are the runtime's to report; state changes, enforcement and
artifacts are recorded by the supervisor alone.

## Run archive

```
experiment.json        the configuration, byte for byte
world-release.json     the world manifest
contracts/v1/          the schemas needed to read this archive
events.jsonl           the record: one canonical JSON event per line, hash-chained
artifacts/index.jsonl  artifact metadata records
artifacts/sha256/..    preserved file contents, by digest
summary.json           final state, counts, world digest
runtime.stderr.log     anything the runtime process wrote to stderr
```

## Layout

```
src/open_lab_simulation/
  supervisor.py   experiment manager and supervisor (the parent process)
  worker.py       agent runtime (the child process) and the agent loop
  toolhost.py     capability invocation, permissions, tool.call / tool.result
  model.py        model client interface and the scripted stand-in
  recorder.py     the single writer of the event log
  artifacts.py    workspace scanning and preservation
  proctree.py     kill a process and all its descendants
  world.py        world release verification (content digest)
  contracts.py    validation against a release's bundled schemas
  archive.py      run archive verification
  cli.py          the `ols` command
experiments/      committed experiment configurations
tests/fixtures/   a world release built by the Builder, used as test data
```

Development rules: [AGENTS.md](AGENTS.md).
