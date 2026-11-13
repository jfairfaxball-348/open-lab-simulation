# AGENTS.md — open-lab-simulation

Instructions for anyone working in this repository: Claude Code, OpenAI Codex,
other coding agents and human contributors. `CLAUDE.md` only imports this file.

## The contract lives in the Builder

This repository is one of three. The development contract for the whole
project — session workflow, honesty rules, the roadmap and the development
state — is **`AGENTS.md` in the sibling `open-lab-builder` checkout**. Read it
first and follow it. Find the sibling checkout next to this one; do not assume
a path.

- The roadmap (`docs/roadmap/ROADMAP.md`) and state (`docs/roadmap/state.json`)
  are in the Builder and are updated there, including for milestones done here.
- The architecture, the contract schemas and the specifications this code
  implements (`docs/architecture/CONTRACTS.md`, `OBSERVATION.md`,
  `SAFETY_AND_ISOLATION.md`) are in the Builder.

What follows is only what is specific to this repository.

## Rules for this repository

- **Never import Builder code** and never copy schemas into `src/`. Schemas are
  read from the world release (`contracts/`) or from a run archive. A test
  enforces this.
- **The recorder has one owner.** Only the supervisor process writes
  `events.jsonl`. The agent runtime and everything below it must never be given
  the archive path or a handle into it.
- **Authority comes only from the experiment configuration.** Nothing in a
  prompt, a model reply or a tool argument may change a limit or a permission.
- **Refuse what cannot be honoured.** If a configuration asks for something
  this runtime does not implement or cannot enforce, the launch is refused.
  Never approximate silently.
- **Record observations, not interpretations.** No event may state what an
  agent wants, feels or believes. Stated goals are `agent.statement` events
  with `epistemic_status: agent_claim`.
- **No real model in the test suite.** Tests use the scripted stand-in. Checks
  against a real model are separate, documented manual steps.
- Capabilities that execute agent-supplied code need the container isolation
  tier. Do not add a way to run them on the process tier.
- `tests/fixtures/worlds/` holds a release built by the Builder. Do not edit it
  by hand; rebuild in the Builder and copy it again when the world changes.
- Never commit run archives, workspaces, personal paths, user or host names,
  credentials or model weights.
- No history rewriting, force pushes or `--no-verify`.

## Commands

| Command | Purpose |
| --- | --- |
| `uv sync` | Create or update the virtual environment |
| `uv run pytest` | Run the tests |
| `uv run ruff check .` | Lint |
| `uv run ols run <experiment.json> [--data-root DIR]` | Launch and supervise one run |
| `uv run ols verify <archive>` | Check a run archive |

## Code conventions

Python 3.12, standard library first (`jsonschema` is the only runtime
dependency). Must run on Windows and Linux: `pathlib`, argument vectors, never
a shell string. Every behaviour has a test. Comments explain constraints and
reasons.
