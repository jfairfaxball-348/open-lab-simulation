"""Offline S012 readiness check. This file cannot launch a run or request inference."""
import hashlib
import json
from pathlib import Path

from open_lab_simulation import chatgpt, policies, providers  # noqa: F401
from open_lab_simulation.paths import data_root
from open_lab_simulation.supervisor import _unsupported, continuation_message
from open_lab_simulation.world import load_world

HERE = Path(__file__).resolve().parent


def digest(value):
    return hashlib.sha256(value).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def verify_plan(directory, world):
    frozen = json.loads((directory / "freeze.json").read_text(encoding="utf-8"))
    raw = (directory / "first-observation.json").read_bytes()
    prompt = (directory / "initial-prompt.txt").read_bytes()
    if digest(raw) != frozen["config_sha256"] or digest(prompt) != frozen["prompt_sha256"]:
        raise ValueError("frozen configuration or approved prompt bytes changed")
    config = json.loads(raw)
    if config["initial_prompt"].encode("utf-8") != prompt:
        raise ValueError("configuration does not contain the exact approved prompt")
    if config["world"]["content_digest"] != world.manifest["content_digest"]["value"]:
        raise ValueError("Habitat release differs from the configured world")
    problems = world.contracts.validate(config, "experiment-config")
    unsupported = _unsupported(config, world)
    if problems or unsupported:
        raise ValueError("unsupported or invalid launch configuration")
    policies.resolve(config, continuation_message(config, world))
    catalog = config["extensions"]["model_catalog"]
    if digest(canonical(catalog["document"])) != catalog["sha256"]:
        raise ValueError("model catalog pin does not match its document")
    return config, frozen


def check(root):
    world = load_world(root / "worlds/habitat/1.0.0")
    config, frozen = verify_plan(HERE, world)
    if (root / "runs" / config["experiment_id"]).exists():
        raise ValueError("S012-001 already has an allocated run; no repeat is authorised")
    checked = providers.launch_check(root, "openai-chatgpt")
    profile = checked["profile"]
    if digest(canonical(profile)) != frozen["provider_profile_sha256"]:
        raise ValueError("broker profile changed; review before any launch")
    used = providers.Ledger(root, "openai-chatgpt").usage("", None)
    if any(used[key] + frozen["max_requests"] > profile["limits"][key]
           for key in ("per_day", "total")):
        raise ValueError("existing broker capacity cannot cover the proposed allowance")
    if profile["limits"]["per_run"] != frozen["max_requests"]:
        raise ValueError("broker per-run ceiling differs from the proposed allowance")
    return config


if __name__ == "__main__":
    try:
        check(data_root())
    except (OSError, ValueError, providers.ProviderError) as error:
        raise SystemExit(f"S012 readiness refused: {error}") from None
    print("Offline S012 checks passed. No run or model request was started.")
    print("Launch still requires the owner's new-chat run next session instruction.")
