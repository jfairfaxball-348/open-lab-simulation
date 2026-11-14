# The Open Laboratory — Simulation

Runs one experiment: a single initial prompt, an isolated workspace, an agent
loop under a supervisor, and a record of everything observable that happened.

This is one of three independent repositories. The **World Builder**
(`open-lab-builder`) engineers the world and owns the architecture, contracts
and roadmap. The **Window** (`open-lab-window`) is a read-only view of what
this repository records. They share contract schemas and a data directory,
never code.

## Status

S005a refresh: tests and experiment configurations use Builder-built minimal
world 0.4.0, including its generated registry and documentation, plus
`workspace.write`, `workspace.list` and `time.now`. The fixture is copied from
the built release, never edited by hand. Git line-ending conversion is disabled
for world fixtures so every digest-bound byte survives checkout. S005b updates
the fixture and experiment configurations to minimal world 0.5.0, which bundles
observation-event 1.2.0 for the recorded discovery preamble.

**S005b discovery is implemented.** Only `capabilities.search`, `.describe`,
`.demonstrate` and `.invoke` are offered to the model. Search reads the world's
registry; describe returns full documentation, the input schema and an example
call; demonstrations execute real code and can change the workspace. Every
interface enforces the experiment's capability restrictions and permissions.
Direct capability calls are refused. The generic system preamble names no
world capability and is recorded before the first model call.

`uv run ols run experiments/scripted-discovery.json` exercises all four tools
without a model. `experiments/discovery-check.json` is the bounded local-model
acceptance check; its initial prompt contains no tool names. The real-model
sequence passed with recorded argument repairs. The final check explicitly guides
argument structure, and earlier failed trials are retained. This does not establish reliable autonomous
tool use. See the Builder's `docs/experiments/2026-11-14-discovery-meta-tools.md`.

`uv run ols run experiments/scripted-workspace.json` performs a short scripted
file-writing check with the shipped capabilities. Its real archive and text
artifact were checked in the Window during S005a; details live in the Builder's
`docs/experiments/2026-11-14-registry-artifacts.md`.

**S003a (runtime skeleton) and S003b (local model) are complete.** What exists
and works:

- A chat model client for a local Ollama server (OpenAI-compatible endpoint)
  with tool calling, repair of malformed tool calls, one retry, and a
  context-length guard. The exact model (digest, server version, sampling) is
  written to `model.json` in the archive. `ols measure-model` reports a
  model's memory, load time and speed.

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

- **No experiment has been run.** One small model (`llama3.2:3b`) has driven
  short engineering checks, and three more candidates were compared in
  S003c. See the Builder's `docs/experiments/`.
- A reply without a tool call does not end a run: the runtime answers with
  the experiment's continuation message (recorded as an `environment.message`
  event) and the run continues to its step limit or deadline. Worlds built
  before 0.3.0 are refused because their contracts cannot record this.
- The scripted stand-in remains for tests: it replays fixed replies and its
  events carry the model id `scripted`. The test suite never calls a model.
- One model per run, one provider kind (Ollama on this machine). The context
  window is whatever the server uses by default.
- No memory, checkpoints, crash recovery, resource
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

With a model pulled into Ollama and world `minimal` 0.2.0 published:

```bash
uv run ols run experiments/local-model-check.json
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
       ├─ model client  → scripted stand-in, or a local model server (loopback only)
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
model.json             exactly which model drove the run
runtime.stderr.log     anything the runtime process wrote to stderr
```

## Layout

```
src/open_lab_simulation/
  supervisor.py   experiment manager and supervisor (the parent process)
  worker.py       agent runtime (the child process) and the agent loop
  toolhost.py     capability invocation, permissions, tool.call / tool.result
  model.py        model client interface, scripted stand-in, chat model client
  ollama.py       model identity and measurement through Ollama's own API
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
