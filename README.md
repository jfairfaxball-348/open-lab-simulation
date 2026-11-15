# The Open Laboratory — Simulation

Runs one experiment: a single initial prompt, an isolated workspace, an agent
loop under a supervisor, and a record of everything observable that happened.

This is one of three independent repositories. The **World Builder**
(`open-lab-builder`) engineers the world and owns the architecture, contracts
and roadmap. The **Window** (`open-lab-window`) is a read-only view of what
this repository records. They share contract schemas and a data directory,
never code.

## Status

S008c engineering acceptance uses the generated minimal 0.8.0 fixture containing
`stats.describe` and `stats.regress`. Discovery, demonstration, invocation,
invalid-input/degenerate cases and deny lists are tested through the supervisor;
the statistics capabilities need no filesystem/network permission. A separate
`experiments/s008c-scripted.json` archive verifies without any real model call.
The four-cell hosted check is prepared in `campaigns/s008c-001.json` and
`scripts/s008c_check.py`; its first launch requires a recorded owner confirmation
of quota availability (`--quota-clearance`). No S008c hosted request has been
made, and the milestone remains incomplete. Earlier 0.6.0/0.7.0 fixtures remain
unchanged. Runtime, provider, context, continuation and repair code are unchanged.

S007e adds supervisor-mediated `models.ask` in minimal world 0.7.0. A hosted
orchestrator can consult the approved installed local llama3.2:3b through one
bounded lease. The runtime receives text, never a model-server address. The
closed `extensions.specialist` configuration sets prompt/output/token/time,
lease and per-run call ceilings. Permission checks still require GPU and
`specialist-text`; the capability itself has no network/filesystem permission.
The supervisor records/reserves calls before dispatch, monitors deadlines during
inference, unloads on segment exit/expiry and reacquires on resume without
replenishing the call budget. It refuses other resident models. Finite native
keep-alive limits crash residency; it is not hardware isolation. 0.6.0 remains
unchanged. See Builder ADR-0018 and CONTRACTS.md; the bounded capability check
is frozen in `campaigns/s007e-001.json` and run with
`uv run python scripts/s007e_check.py` (one next cell, at most four total).
The check is now closed with partial coverage after a provider plan-limit stop:
guided and C1 finished, C2 was censored, C3 was not started. Guided consultation
and use of the returned answer worked; unguided specialist use is not established.
Builder's `docs/experiments/2026-11-15-specialist-consultation.md` records the
results and owner-directed partial closeout. The runner refuses further cells
after the recorded provider stop; it does not silently restart the check.

**S006a memory and checkpoints are implemented.** Minimal world 0.6.0 adds
`memory.write`, `memory.search` and `memory.list`. Agent-owned SQLite data lives
at `.memory.sqlite` in the workspace; `NOTES.md` is the optional working-notes
convention. No permission or supervisor state is read from either file.

Enable `checkpointing` in an experiment to take an initial snapshot, interval
snapshots at completed-step barriers, and a checkpoint before normal completion
or cooperative operator pause. Defaults: interval 60 seconds, keep 3. The
supervisor snapshots conversation, scripted cursor/counters, SQLite memory and
the rest of the workspace into protected, digest-bound checkpoint directories.

```bash
uv run ols run experiments/scripted-memory-checkpoint.json --pause-after-steps 2
uv run ols verify <archive printed above>
uv run ols resume <archive printed above>
uv run ols verify <archive printed above>
```

Resume verifies the archive and selects the newest valid, unpruned checkpoint,
falling back if one is corrupt. It appends to the same log with a recovery
event; existing observations are never erased. Files, memory and conversation
are restored together, and all consumed steps still count against the original
limit. Active time accumulates; paused downtime does not. Changed world/model
digests, exhausted budgets, live/stale writer locks and invalid logs are refused.
The previous workspace is retained beside the restored one to preserve operator
edits. Prior summaries are kept under `segments/`; stderr is appended.

Corrupt snapshots remain forensic evidence and `ols verify` still reports them
after a successful fallback. Hard deadlines retain the preceding safe snapshot.

**S006b graceful deadlines are implemented.** Set
`termination.on_deadline` to `checkpoint_then_stop` with `checkpointing` enabled.
At the original deadline the supervisor records a stop request. The in-flight
step may finish within `termination.grace_period_s` (default zero), then the
supervisor snapshots at its complete-step barrier and kills the tree before
acknowledging it. No following step starts. Grace is one fixed allowance after
the original deadline, including snapshot time; it is not reset by resume.
A hung model/capability or expired snapshot allowance causes termination using
the preceding safe checkpoint. `hard_stop` ignores grace and retains its
original behaviour. Active time includes grace, so a run that exhausts its
original budget cannot be manually resumed to obtain more time.

```bash
uv run ols run experiments/scripted-graceful-deadline.json
uv run ols verify <archive printed above>
```

**S006d recovery is implemented.** It is opt-in through the frozen experiment:

```json
"checkpointing": {"interval_s": 60, "keep_last": 3},
"extensions": {
  "recovery": {
    "max_restarts": 2,
    "auto_resume": true,
    "health_check_interval_s": 5,
    "health_check_timeout_s": 2
  }
}
```

`max_restarts` is required (0–100); the other fields above are defaults except
`auto_resume`, which defaults to false. The long-lived supervisor restarts a
crashed worker from the newest valid checkpoint. Restarts share one persisted
retry budget across segments and startup scans. Every model attempt is reserved
and synced before it starts; killed attempts still consume the original step
budget. A host-local accounting ledger reserves one second of active time ahead
and renews every 0.2 seconds. Expired/failed accounting stops the worker. After
an interruption, unused reservation time may be charged conservatively;
wall-clock downtime is not added. Normal resume uses actual accumulated active
time and also charges revalidation/restoration work.

For real models, read-only `/api/tags` probes check server availability and the
original model digest. Probes run off the deadline thread and have a supervisor
timeout. A failed probe stops the tree and enters the same bounded recovery
path. Revalidation can refuse an unavailable or changed model; no model is
pulled or server started automatically. A healthy identity probe does not prove
that inference will succeed.

```bash
uv run ols run experiments/scripted-recovery.json
uv run ols recover
```

`ols recover` is the startup scan to run after restarting the supervisor/host.
It resumes only incomplete archives with `auto_resume: true`. This change does
not install an OS startup task. Completed/paused/stopped runs are not replayed.
A permanent OS-locked `.writer.guard` serializes writers; `.writer.lock` records
a hashed machine identity, PID and OS process birth identity. A live writer is
never displaced. Only a confirmed absent/reused owner on this machine may have
its stale marker archived under `segments/`. Legacy, malformed, foreign-machine
or inaccessible ownership is refused. The old runtime must also be absent.
On Linux a surviving old process group causes refusal.

Recovery preserves every existing event and artifact version. Missing summaries
can be recovered from complete valid logs and valid accounting/checkpoints.
Partial logs, invalid accounting and inconsistent artifacts are refused without
trimming observations. Refusals append supervisor actions only when exclusive
ownership and a valid log make that safe; otherwise the CLI reports refusal
without changing the record. Physical reboot/power-loss durability, Linux and
large-run/uninterruptible host-I/O timing are unverified. See the Builder's
ADR-0011 and `docs/experiments/2026-11-14-supervisor-recovery.md`.

**S006e resources and sampling are implemented.** Experiment
`limits.resources` accepts `ram_mb`, `max_processes`, `cpu_cores` and `disk_mb`:

```json
"limits": {
  "wall_clock_s": 20,
  "resources": {"ram_mb": 256, "max_processes": 8, "cpu_cores": 1, "disk_mb": 1}
}
```

On Windows, RAM is the Job Object's aggregate committed-memory ceiling, process
count includes Python/venv launchers and capabilities, and CPU is a hard cycle
rate `floor(cpu_cores / logical_processors * 10000)` out of 10000. The runtime
starts suspended, joins the configured job, then resumes, so the venv redirector
cannot spawn outside it. Limits are installed again on each recovery/resume.
CPU settings outside the host's representable range and hosts with more than
64 logical processors are refused. CPU active-time samples are a different
measurement from the kernel cycle budget; short intervals need not equal the
nominal core-equivalent rate. RAM/process-count limit notifications cause tree
termination and a supervisor action. Windows notification delivery is best
effort; the OS ceilings remain in effect even if a notification is missed.

`disk_mb` is a supervisor workspace quota, checked at writing-capability
admission and every 0.1 seconds while supervising. At the quota, all further
writing capabilities (including demonstrations) are denied; reads remain
available. An active writer observed above quota is killed with the runtime
tree. A completed write may overshoot before the next check, then future writes
are denied. This is not an OS filesystem quota. It counts logical file bytes,
including memory/SQLite sidecars, and excludes the archive and model files.
Unreadable/link-containing workspaces fail closed. A conservative denial latch
lasts for that segment. Resource termination does not automatically restart.

Periodic supervisor `resource.sample` events default to one second; override
with `extensions.resource_sampling.interval_s` (finite 0.05–60 seconds). CPU is
100 percent per logical core from job accounting, RAM is aggregate process
private commit, process count is job accounting, and workspace disk is logical
file size. First CPU samples or racing/inaccessible process-memory queries may
be unavailable. Missing numeric values are omitted and named in
`payload.extensions.unavailable`; observed zero remains numeric zero.

When existing `nvidia-smi` is usable, an asynchronous two-second-bounded read
samples aggregate host NVIDIA VRAM used/available. Samples record its scope and
age (up to ten seconds); it is not attributed to this run. The sampler never
installs a driver/tool or blocks the deadline thread. Model servers and the
supervisor live outside the runtime job. `vram_mb` limits (even zero) and nested
resource `wall_clock_s` are refused; use the experiment's top-level deadline.
`max_model_tokens` is also explicitly refused until it can be enforced. On
non-Windows platforms requested CPU/RAM/process-count limits are refused;
missing job measurements are marked unavailable. Linux execution is unverified.

```bash
uv run ols run experiments/scripted-resources.json
uv run ols verify <archive printed above>
```

See Builder ADR-0012 and `docs/experiments/2026-11-14-resource-enforcement.md`.
Manual resume normally needs a
stopped run with a summary; a failed recovery-enabled run cannot bypass the
remaining retry budget by manual resume. Deadline notices are still refused. Snapshot deadline
checks are cooperative between I/O chunks; uninterruptible host I/O and
large-run timing bounds are unverified. See the Builder's ADR-0010.
Operator `--pause-after-steps` is the tested CLI stop mechanism. Ctrl+C while
the supervisor waits requests a pause at the next complete-step barrier; native
console signal delivery to the entire process tree remains unverified.
See the Builder's `docs/decisions/ADR-0009-checkpoint-barriers-and-agent-memory.md`
and `docs/experiments/2026-11-14-memory-checkpoints.md`.

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

**S007d-001 is complete: NO_GO.** All 24 registered cells and archives verified,
with 24 source-bound reviews. A completed the full guided task in 0/6 runs;
B/C each had 0/6 qualifiers. Real partial tool use and two exposed fault-response
examples are retained; they do not open expansion. See Builder
`docs/experiments/2026-11-14-autonomy-baseline.md` and
`docs/experiments/results/S007d-001.json`.

**ChatGPT plan-usage adapter (S014d) is implemented and tested against fakes
only.** `ols provider sign-in openai-chatgpt` runs OpenAI's documented Sign in
with ChatGPT flow in the owner's own browser (dynamic registration, PKCE,
loopback callback on 127.0.0.1, RS256 ID-token check, plan-usage scope check)
and stores the tokens with DPAPI; `sign-out` revokes the refresh token and
clears them while keeping the registration and host id; `models` lists the
account's models. The adapter (`openai-responses-chatgpt-plan` 1.0.0) sends
stateless Responses requests (`store: false`, `stream: true`, tools in one
namespace, `strict: false`), refreshes tokens near expiry, never sends
temperature, top-p, output cap or seed (recorded as differences), and never
resends or records reasoning items. Live checks (S014d, S014g) completed a
tool-use round trip on `gpt-6.1-sol`. `tests/test_hosted_compatibility.py` is
the S014g admission suite: 18 full supervised runs against a fake provider.

**Hosted provider broker (S014b) is implemented.** Adapters exist for Google
Gemini (S014c) and ChatGPT plan usage (S014d).
A world or catalog model whose provider has `locality: external` is reached only
through a loopback broker inside the supervisor (ADR-0016): the runtime sends its
usual chat-completions requests with a per-run token, and the broker alone reads
the provider credential (Windows DPAPI, under the user profile), reserves budget
in `<data root>/providers/<id>/ledger.jsonl` before every attempt (retries
included; unknown outcomes count as spent), sends one request at a time and
mirrors each request into the run's `provider-ledger.jsonl`. Launch is refused
unless the provider is implemented, enabled, budgeted and credentialed; hosted
runs refuse automatic recovery; nothing falls back to another model. Owner
administration: `ols provider status|set-key|clear-key|configure <id>` (the key
is typed at a hidden prompt, never passed as an argument), plus
`sign-in|sign-out|models openai-chatgpt`.

**S007h-001 is complete: NO_GO.** The same frozen design with one change, the
installed `qwen3:4b` as orchestrator (`campaigns/s007h-001.json`; runtime source
unchanged). 24/24 cells verified and reviewed; every cell reached the 600 s
deadline after 7-21 calls; 0/6 guided and 0/6 B/C qualifiers; seven successful
invocations and zero repairs. See Builder
`docs/experiments/2026-11-14-model-swap-reevaluation.md`. The next route is a
registered decision (OD-9); no further campaign or improvement is implemented.

**Campaign v1.1 and rubric v1.1 (S014m).** A manifest of version
`s007d-campaign-v1.1` is scored with rubric v1.1 (Builder
`docs/experiments/S007d-RUBRIC-v1.1.md`): the guided-control report may be the
returned note exactly, trimmed, or as a JSON string body, and a run ended by a
provider usage-limit refusal is labelled `PROVIDER_CENSORED` (INCONCLUSIVE only
when censored cells could have changed the gate decision). v1.1 manifests carry
their own seed labels (evaluation below 900, pilots 900 and above) and the
provider-stop rule: after a cell that failed or whose provider use ended
terminally, the runner records `provider_stop` and starts no further cell until
it is run with `--clearance "<operator note>"`, which is recorded in the ledger.
v1 manifests, their pins and their scoring are unchanged. All recorded JSONL is
read on `"\n"` only (S014l), so text holding U+2028/U+2029/U+0085 verifies.

**S007d benchmark tooling is implemented.**
`python -m open_lab_simulation.benchmark run <manifest> --registration <commit>`
executes at most four scheduled cells per invocation (override `--max-cells`).
Evaluation manifests require all 24 cells: seeds 41/42/43, two replicates and
conditions A/B/C/D, in reproducibly randomised balanced blocks. Pilot seeds and
roles are separate. Prompt/profile/config/source fingerprints are pinned before
launch; source fingerprints normalise Python CRLF/LF, matching Git's text rules.
A verified OS writer lock serialises runners. The append-only, fsynced ledger
records starts, finished cells, failed launches and reconciled interruptions.
Started cells are never silently rerun or replaced. Results stay outside the
agent workspace under the data root's `campaigns/<id>/` directory.

The D arm injects one explicitly marked transient failure at the first actual
capability execution after permission/schema/interface checks and before side
effects. Invalid/denied calls and discovery do not consume it; demonstrations
can. Supervisor injection records and delivered tool results distinguish the
benchmark fault from a real capability failure. Exposure requires a subsequent
recorded model input containing the failure. No instruction/tool suggestion is
added, no privilege is granted and restarts cannot replenish the injection.

`python -m open_lab_simulation.benchmark report <manifest>` verifies archives,
conditions, fresh initial workspace/memory and source-bound investigator
annotations. Metrics distinguish observations, agent claims and investigator
assessments. Annotations live under `campaigns/<id>/annotations/`, outside raw
archives, and bind the exact event-log hash. Pilot reports are `NOT_SCORED`;
evaluation reports separately return GO/NO_GO/INCONCLUSIVE. Unknown settings and
usage, resource exhaustion, failed controls, unexposed faults and censored runs
remain visible. No autonomy score or internal-motivation inference is produced.
See the Builder's S007d preregistration/rubric/report for the scientific scope.

**S007c runtime policies are implemented.** The optional
`extensions.runtime_policy` selects `legacy-immediate-v1` (default) or
`quiet-until-deadline-v1`. Quiet stops model calls after a configured consecutive
tool-free reply threshold (default 1, maximum 16); below it, the fixed recorded
continuation permits bounded reflection opportunities. At the threshold no new
text is appended and no automatic wake occurs. Waiting uses the remaining active
wall-clock budget. Pause/resume and recovery preserve the idle latch and counters;
step ceilings still stop immediately. Prose never controls scheduling authority.

`policy.json` freezes resolved policy/defaults and the experiment digest, bound
to the launch log. Context uses version `oldest-tools-turns-v1`, explicit memory
uses `explicit-tools-v1` and turn semantics use `user-continuation-v1`; unknown
versions/fields/semantics are refused. `context_guard_tokens` sizes the existing
client guard and never promises server context. Initial prompt, preamble and
newest assistant/tool group survive; protected overflow is reported. Memory notes
are not automatically inserted, retrieved or summarised.

Before each HTTP attempt a supervisor `model_input` action records the complete
request (messages, tools, requested settings) and context mutations, then
acknowledges it before the worker contacts the server. Retries and failed calls
retain inputs. Scripted runs record their ignored input separately. Scheduling
barriers preserve checkpoint safety while idle; they do not poll inference.
Raw request snapshots can grow archives significantly; long-run overhead is
unverified. Historical archives keep their original records; new recovery
segments record their legacy interpretation explicitly.

```bash
uv run ols run experiments/s007c-legacy.json
uv run ols run experiments/s007c-quiet.json --pause-after-steps 1
uv run ols resume <archive printed above>
```

These are directed engineering pilots with an installed model. They do not
run S007d or establish neutrality/autonomy. See Builder
`docs/architecture/RUNTIME_POLICIES.md` and its S007c engineering report.

**S007a local gateway is implemented.** Optional
`extensions.model_catalog` selects one approved model-provider document outside
the world. It contains `document` and `sha256`: the SHA-256 of the document's
canonical JSON (sorted keys, compact separators, literal Unicode, UTF-8, no
trailing newline). Both paired configurations embed the same Builder-owned
catalog; only experiment identity and the orchestrator selection differ.
Bundled manifests remain the default. Catalog selection grants no permissions.

```bash
uv run ols run experiments/s007a-swap-llama.json
uv run ols run experiments/s007a-swap-granite.json
```

The gateway accepts local Ollama and loopback chat-completions servers
(`openai_compatible` or `llama_cpp`); it refuses credentials in URLs, redirects,
external hosts, unavailable models, duplicate ids and mismatched catalog/model
digests. Generic servers must list the exact model through `/models`; their
weight identity is explicitly unknown, so checkpoint resume is refused without
an attestable weight digest. Ollama continues using its native identity API.
No real generic server was tested; adapter tests use a fake.

`model.json` freezes provider/model manifests, adapter version, requested
sampling, client context guard and repair policy. Effective sampling, context
and offload are explicitly unknown rather than inferred from the manifest.
`model-catalog.json` preserves canonical catalog bytes. A hash in the launch
state event binds model provenance; `ols verify` checks both files. Resume
revalidates pinned catalogs/settings and model identity. Old archives remain
verifiable. The default output request is now explicitly 512 tokens when omitted.
Server resources stay outside runtime job limits. S007e adds specialist consultation
and one GPU lease; ChatGPT plan usage is enabled only under the owner's approved
profile and milestone budgets. Other hosted routes remain separately gated.

What does **not** exist yet:

- **The registered S007d-001 baseline is complete with NO_GO.** Earlier local
  runs remain engineering checks. Long philosophical-duration experiments and
  expanded-capability campaigns remain future work; see Builder reports.
- A reply without a tool call does not end a run: the runtime answers with
  the experiment's continuation message (recorded as an `environment.message`
  event) and the run continues to its step limit or deadline. Worlds built
  before 0.3.0 are refused because their contracts cannot record this.
- The scripted stand-in remains for tests: it replays fixed replies and its
  events carry the model id `scripted`. The test suite never calls a model.
- One orchestrator per run; local provider routing is implemented in S007a.
  The context window remains server-controlled and must be measured separately.
- The *container tier* (S006c) exists for Windows hosts only: a world with
  `minimum_tier: container` runs every capability call in a fresh rootless
  Podman container in WSL 2 (pinned base image, verified world read-only, the
  workspace as the only host mount, no network, no Linux capabilities, read-only
  root, memory/CPU/process ceilings). It launches only when the backend probe
  passes; containerised capabilities cannot be granted network or GPU. The
  backend is recorded in `isolation.json`. No shipped capability runs
  agent-supplied code yet, and agent-created tools are refused.
- Worlds on the *process tier* remain an architectural boundary for
  capabilities written by the Builder, not a sandbox.
- Process-tree termination and the container tier have only been run on
  Windows. The container escape/limit tests skip when the backend is absent;
  `OPEN_LAB_REQUIRE_CONTAINER=1` makes them fail instead.

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
  recovery.py     opt-in policy, active-time ledger and asynchronous health probes
  resources.py    supervisor quota admission, periodic samples and GPU probe
  ownership.py    OS writer guard and verified process birth identities
  worker.py       agent runtime (the child process) and the agent loop
  toolhost.py     capability invocation, permissions, tool.call / tool.result
  policies.py     versioned continuation/context/memory treatment and state validation
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

## Licence

Apache License 2.0; see [LICENSE](LICENSE).
