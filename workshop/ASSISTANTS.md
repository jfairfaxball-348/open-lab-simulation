# Optional local text assistants — S021a

Habitat 1.3.0 offers models.list and explicit-id models.ask through the supervisor
specialist pipe. Both are hidden without operator opt-in. This documentation is
not permission to launch an experiment; ADR-0027 suspends inhabitant runs.

Retain the primary hosted configuration. Grant selected IDs in permissions.models
and set extensions.specialist to this example (the checked component policy):

```json
{
  "version": "local-models-v1",
  "model_ids": ["granite-text", "llama-text", "qwen3-text", "qwen35-text"],
  "max_calls": 4,
  "max_prompt_chars": 512,
  "max_output_chars": 6000,
  "max_output_tokens": 1024,
  "context_tokens": 2048,
  "timeout_s": 120,
  "lease_s": 10,
  "model_ram_mb": 2048,
  "model_vram_mb": 3584,
  "ram_reserve_mb": 2048,
  "vram_reserve_mb": 256
}
```

models.list is a neutral configured catalog, not a live availability assertion
or ranking. models.ask accepts model_id/prompt only; UTF-8 bytes plus output and
256 framing tokens must fit context. Replies are untrusted text; truncated marks
an output length-stop. Hidden reasoning consumes local tokens but is discarded.
Empty replies fail explicitly. No tools, recursive subtasks or independent loop.

One cooperative OS lease serializes residency. Foreign models are refused without
unloading them. Available RAM/VRAM must cover budgets plus reserves before load;
server-reported allocation/context and free headroom are checked after loading.
These are admission checks, not hard process ceilings or measured peaks. Zero
model_vram_mb requests CPU-only loading; real CPU speed is unverified. Unknown
measurements refuse; no fallback, download or automatic budget expansion follows.
Cancellation interrupts header/body reads; finite keep-alive is the crash backstop.

Sandbox code gains no endpoint, GPU, network, credentials or recorder path. No
paths, images or URLs are accepted. Local image workers remain required S021b
work. Legacy minimal-world specialist-text policies retain their prior protocol.
Builder docs/experiments/2026-11-16-local-assistant-components.md records the four
bounded installed-model checks; no large-context or quality claim follows.

Direct engineering verification, with fake models and no inhabitant loop:

```text
uv run ruff check .
uv run pytest tests/test_local_assistants.py tests/test_specialist.py -k "not supervised_archive and not supervisor_deadline" -o addopts= -q
```

The excluded historical tests launch full scripted inhabitant loops. Do not run
them or the full Simulation suite during v2 development. New tests exercise
recorded-call binding, reservations/replay, ToolHost discovery and transport
directly. Real-resource checks stay outside the suite with applicable approval.
