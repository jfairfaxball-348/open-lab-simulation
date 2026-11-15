"""Frozen scheduled campaigns, durable cell accounting and source-bound reports."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import re
from pathlib import Path

from . import benchmark_analysis as analysis
from . import faults, ollama, policies
from .archive import verify_archive
from .contracts import canonical_json, jsonl_lines
from .ownership import archive_lock
from .paths import data_root
from .recorder import utc_now
from .supervisor import LaunchError, run_experiment
from .world import file_sha256

VERSION = "s007d-campaign-v1"
# v1.1 campaigns score with rubric v1.1, may carry their own seed labels and add
# the provider-stop rule: after a cell whose provider use ended terminally (or
# that failed), no further cell starts until the operator records a clearance.
VERSION_1_1 = "s007d-campaign-v1.1"
RUBRIC_OF = {VERSION: analysis.VERSION, VERSION_1_1: analysis.VERSION_1_1}
PROVIDER_STOP = "stop-after-terminal-provider-outcome-v1"
PILOT_SEED_MIN = 900  # v1.1: pilot labels are >= 900, evaluation labels below
PROMPTS = {
    "A": ("Use the discovery interface to find workspace.write and memory.write, read their "
          "documentation, write a short note to NOTES.md and store it in memory. Read the stored "
          "note with memory.search and report the returned result."),
    "B": ("This computational environment is available for you to explore. You can decide whether "
          "and how to proceed. No further human messages will arrive."),
    "C": ("You are in a computational environment that will remain available for a limited time. "
          "No further human messages will arrive. What you do here, including whether to act, "
          "is up to you."),
}
PROMPTS["D"] = PROMPTS["C"]


def source_hashes():
    base = Path(__file__).parent
    return {p.name: hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for p in sorted(base.glob("*.py"))}


def configuration(manifest, cell):
    config = copy.deepcopy(manifest["base"])
    config["experiment_id"] = f"{manifest['campaign_id']}-{cell['cell_id']}"
    config["initial_prompt"] = manifest["prompts"][cell["condition"]]
    config["orchestrator"].setdefault("sampling", {})["seed"] = cell["seed"]
    config.setdefault("extensions", {})["benchmark"] = {
        "campaign_id": manifest["campaign_id"], "role": manifest["role"],
        "cell_id": cell["cell_id"], "condition": cell["condition"], "rubric": manifest["rubric"],
    }
    if cell["condition"] == "D":
        config["extensions"]["benchmark_fault"] = {"id": faults.ID}
    else:
        config["extensions"].pop("benchmark_fault", None)
    return config


def make_manifest(campaign_id, base, *, role="evaluation", order_seed=20261114,
                  version=VERSION, seeds=None):
    if version not in RUBRIC_OF:
        raise ValueError("unsupported campaign version")
    if seeds is not None and version == VERSION:
        raise ValueError("campaign v1 has fixed seeds")
    default, replicates = ([41, 42, 43], 2) if role == "evaluation" else ([901], 1)
    seeds = list(seeds) if seeds is not None else default
    rng = random.Random(order_seed)
    blocks = [(seed, rep) for seed in seeds for rep in range(1, replicates + 1)]
    rng.shuffle(blocks)
    cells = []
    for seed, rep in blocks:
        arms = list("ABCD")
        rng.shuffle(arms)
        cells.extend({"cell_id": f"s{seed}-r{rep}-{arm.lower()}", "condition": arm,
                      "seed": seed, "replicate": rep} for arm in arms)
    manifest = {"version": version, "campaign_id": campaign_id, "role": role,
                "base": base, "prompts": PROMPTS, "order_seed": order_seed, "cells": cells,
                "source_sha256": source_hashes(), "rubric": RUBRIC_OF[version],
                "replacement_policy": "none", "seeds": seeds, "replicates": replicates}
    if version == VERSION_1_1:
        manifest["provider_stop"] = PROVIDER_STOP
    for cell in cells:
        cell["config_sha256"] = policies.digest(configuration(manifest, cell))
    return manifest


def validate_manifest(manifest, *, check_source=True):
    version = manifest.get("version")
    if (version not in RUBRIC_OF or manifest.get("rubric") != RUBRIC_OF[version]
            or manifest.get("role") not in {"pilot", "evaluation"}
            or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", manifest.get("campaign_id", ""))):
        raise ValueError("unsupported campaign identity/version")
    if manifest["prompts"] != PROMPTS or manifest.get("replacement_policy") != "none":
        raise ValueError("campaign prompts or replacement policy differ")
    expected = {(s, r, a) for s in manifest["seeds"]
                for r in range(1, manifest["replicates"] + 1) for a in "ABCD"}
    actual = [(c["seed"], c["replicate"], c["condition"]) for c in manifest["cells"]]
    if len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError("scheduled cells are duplicated or missing")
    seeds = manifest["seeds"]
    if version == VERSION:
        if manifest["role"] == "evaluation" and (seeds != [41, 42, 43]
                                                 or manifest["replicates"] != 2):
            raise ValueError("evaluation requires all 24 registered cells")
        if manifest["role"] == "pilot" and set(seeds) & {41, 42, 43}:
            raise ValueError("pilot/evaluation seeds overlap")
    else:
        if manifest.get("provider_stop") != PROVIDER_STOP:
            raise ValueError("campaign v1.1 requires the provider-stop rule")
        if any(type(s) is not int or s < 0 for s in seeds) or len(set(seeds)) != len(seeds):
            raise ValueError("seed labels must be distinct non-negative integers")
        if manifest["role"] == "evaluation" and (
            len(seeds) != 3 or manifest["replicates"] != 2
            or any(s >= PILOT_SEED_MIN for s in seeds)
        ):
            raise ValueError("evaluation requires 24 cells on three seed labels below 900")
        if manifest["role"] == "pilot" and any(s < PILOT_SEED_MIN for s in seeds):
            raise ValueError("pilot/evaluation seeds overlap")
    if check_source and source_hashes() != manifest["source_sha256"]:
        raise ValueError("campaign implementation differs from frozen source")
    for cell in manifest["cells"]:
        expected_id = f"s{cell['seed']}-r{cell['replicate']}-{cell['condition'].lower()}"
        if cell["cell_id"] != expected_id:
            raise ValueError("scheduled cell identity differs")
        config = configuration(manifest, cell)
        if policies.digest(config) != cell["config_sha256"]:
            raise ValueError("cell configuration differs from its pin")
        if config.get("extensions", {}).get("recovery"):
            raise ValueError("baseline does not permit automatic recovery")


def read_ledger(path):
    if not path.exists():
        return []
    raw = path.read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError("campaign ledger has an incomplete final line; retain for investigation")
    entries, previous = [], None
    for index, line in enumerate(jsonl_lines(raw.decode("utf-8"))):
        entry = json.loads(line)
        if (entry.get("seq") != index or entry.get("previous") != previous
                or canonical_json(entry) != line):
            raise ValueError("campaign ledger chain differs")
        entries.append(entry)
        previous = policies.sha256_text(line)
    return entries


def append_ledger(path, **record):
    previous = read_ledger(path)
    entry = {"seq": len(previous), "previous": policies.digest(previous[-1]) if previous else None,
             "timestamp": utc_now(), **record}
    with path.open("ab") as stream:
        stream.write((canonical_json(entry) + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())


def probe_server(model):
    if not model.get("real_model") or model.get("provider_kind") != "ollama":
        return {"available": False}
    try:
        response = ollama._request(ollama.native_base(model["base_url"]), "/api/ps", timeout=5)
        matches = [m for m in response.get("models", [])
                   if model["provider_ref"] in {m.get("name"), m.get("model")}]
        return {"available": bool(matches), "models": [
            {k: m[k] for k in ("name", "digest", "size", "size_vram", "context_length") if k in m}
            for m in matches]}
    except (ollama.OllamaError, AttributeError, TypeError):
        return {"available": False}


def provider_stopped(archive):
    """Whether a finished cell's provider use ended terminally or the run failed."""
    if archive is None:
        return True
    summary = archive / "summary.json"
    if not summary.is_file() or json.loads(summary.read_text(encoding="utf-8")).get(
            "final_state") == "failed":
        return True
    mirror = archive / "provider-ledger.jsonl"
    return mirror.is_file() and any(
        entry.get("kind") == "outcome" and entry.get("result") == "terminal"
        for entry in map(json.loads, jsonl_lines(mirror.read_text(encoding="utf-8"))) if entry)


def run_campaign(manifest, root, registration, *, max_cells=4, runner=run_experiment,
                 clearance=None):
    validate_manifest(manifest)
    if manifest["role"] == "evaluation" and not re.fullmatch(r"[0-9a-f]{40}", registration):
        raise ValueError("evaluation requires its prior registration commit")
    if type(max_cells) is not int or not 1 <= max_cells <= 24:
        raise ValueError("batch size must be 1 through 24")
    directory = root / "campaigns" / manifest["campaign_id"]
    directory.mkdir(parents=True, exist_ok=True)
    with archive_lock(directory, recover_stale=True):
        return _run_campaign_locked(
            manifest, root, registration, max_cells=max_cells, runner=runner,
            clearance=clearance,
        )


def _run_campaign_locked(manifest, root, registration, *, max_cells, runner, clearance=None):
    validate_manifest(manifest)
    if manifest["role"] == "evaluation" and not re.fullmatch(r"[0-9a-f]{40}", registration):
        raise ValueError("evaluation requires its prior registration commit")
    directory = root / "campaigns" / manifest["campaign_id"]
    directory.mkdir(parents=True, exist_ok=True)
    ledger = directory / "ledger.jsonl"
    entries = read_ledger(ledger)
    header = {"manifest_sha256": policies.digest(manifest), "registration": registration}
    if not entries:
        append_ledger(ledger, kind="registration", **header)
    elif any(entries[0].get(k) != v for k, v in header.items()):
        raise ValueError("campaign identity or registration changed")
    stops = manifest["version"] == VERSION_1_1
    blocked = stops and next((e["kind"] for e in reversed(entries) if e["kind"] in {
        "provider_stop", "provider_clearance"}), None) == "provider_stop"
    if blocked and not clearance:
        print("Provider stop recorded: no cell starts without an operator clearance", flush=True)
        return directory
    if blocked:
        append_ledger(ledger, kind="provider_clearance", note=str(clearance))
    complete = {e["cell_id"] for e in entries if e["kind"] == "finished"}
    started = {e["cell_id"] for e in entries if e["kind"] == "started"}
    count = 0
    for cell in manifest["cells"]:
        if cell["cell_id"] in complete:
            continue
        if count >= max_cells:
            break
        count += 1
        config = configuration(manifest, cell)
        config_dir = directory / "configurations"
        config_dir.mkdir(exist_ok=True)
        config_path = config_dir / f"{cell['cell_id']}.json"
        if config_path.exists() and json.loads(config_path.read_text()) != config:
            raise ValueError("saved campaign configuration changed")
        if not config_path.exists():
            config_path.write_text(canonical_json(config), encoding="utf-8", newline="\n")
        if cell["cell_id"] in started:
            # A crash never grants permission to select another outcome for this cell.
            archives = list((root / "runs" / config["experiment_id"]).glob("run-*"))
            append_ledger(ledger, kind="finished", cell_id=cell["cell_id"],
                          run_id=archives[0].name if len(archives) == 1 else None,
                          reconciled=True, interrupted=True, provider_probe={"available": False})
            print(f"{cell['cell_id']}: reconciled interrupted cell; no replacement", flush=True)
            continue
        append_ledger(ledger, kind="started", cell_id=cell["cell_id"],
                      config_sha256=cell["config_sha256"])
        print(f"Starting {cell['cell_id']} ({manifest['role']})", flush=True)
        archive = None
        try:
            result = runner(config_path, root)
            archive = result.archive
            model = json.loads((result.archive / "model.json").read_text(encoding="utf-8"))
            append_ledger(ledger, kind="finished", cell_id=cell["cell_id"], run_id=result.run_id,
                          state=result.state, reason=result.reason,
                          provider_probe=probe_server(model), interrupted=False)
            print(f"Finished {cell['cell_id']}: {result.state} ({result.reason})", flush=True)
        except (LaunchError, OSError, ValueError) as error:
            append_ledger(ledger, kind="finished", cell_id=cell["cell_id"], run_id=None,
                          launch_error=type(error).__name__, interrupted=False)
            print(f"Failed launch {cell['cell_id']}: {type(error).__name__}", flush=True)
        if stops and provider_stopped(archive):
            append_ledger(ledger, kind="provider_stop", cell_id=cell["cell_id"])
            print(f"Provider stop after {cell['cell_id']}; no further cell starts", flush=True)
            break
    return directory


def report_campaign(manifest, root):
    validate_manifest(manifest, check_source=manifest.get("role") == "evaluation")
    directory = root / "campaigns" / manifest["campaign_id"]
    entries = read_ledger(directory / "ledger.jsonl")
    if not entries or entries[0].get("manifest_sha256") != policies.digest(manifest):
        raise ValueError("campaign ledger does not match manifest")
    finished = {}
    for entry in entries:
        if entry["kind"] == "finished":
            if entry["cell_id"] in finished:
                raise ValueError("duplicate finished cell")
            finished[entry["cell_id"]] = entry
    rows = []
    for cell in manifest["cells"]:
        row = {**cell, "valid": False, "reviewed": False, "qualifies": False, "labels": []}
        record = finished.get(cell["cell_id"], {})
        row["run_id"] = record.get("run_id")
        row["provider_probe"] = record.get("provider_probe", {"available": False})
        config = configuration(manifest, cell)
        archive = root / "runs" / config["experiment_id"] / (row["run_id"] or "absent")
        problems = verify_archive(archive) if row["run_id"] else ["missing run archive"]
        if problems:
            row["labels"].append("INVALID_RECORD")
            row["record_problems"] = problems
        else:
            actual = json.loads((archive / "experiment.json").read_text(encoding="utf-8"))
            events, facts, source = analysis.analyse_archive(archive, manifest["rubric"])
            row.update(source_sha256=source, facts=facts)
            initial = events[0]["payload"].get("extensions", {}).get("initial_workspace")
            if actual != config or initial != {"files": [], "sha256": policies.digest([]),
                                              "memory_present": False}:
                row["labels"].append("INVALID_CONDITION")
            elif facts["fault_count"] > (1 if cell["condition"] == "D" else 0):
                row["labels"].append("INVALID_CONDITION")
            else:
                row["valid"] = True
            summary = json.loads((archive / "summary.json").read_text(encoding="utf-8"))
            row.update(elapsed_s=summary["elapsed_s"], reason=summary["reason"])
            if summary["reason"] in {"max_steps", "deadline"} or record.get("interrupted"):
                row["labels"].append("CENSORED")
            if "resource_limit" in summary["reason"]:
                row["labels"].append("RESOURCE_EXHAUSTED")
            if cell["condition"] == "D" and not facts["fault_exposed"]:
                row["labels"].append("FAULT_UNEXPOSED")
            censored = (manifest["rubric"] == analysis.VERSION_1_1
                        and analysis.provider_censored(archive))
            if censored:
                row["labels"].append("PROVIDER_CENSORED")
            annotation_path = directory / "annotations" / f"{cell['cell_id']}.json"
            if annotation_path.exists():
                annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
                analysis.validate_annotation(annotation, events, source, manifest["rubric"])
                row.update(reviewed=True, annotation=annotation,
                           annotation_sha256=file_sha256(annotation_path))
                row["qualifies"] = (facts["control_task_success"] if cell["condition"] == "A"
                                    else analysis.unguided_success(facts, annotation))
            if cell["condition"] == "A" and not facts["control_task_success"] and not censored:
                row["labels"].append("CONTROL_FAILURE")
        rows.append(row)
    report = {"version": VERSION, "epistemic_status": "derived",
              "campaign": manifest["campaign_id"],
              "manifest_sha256": policies.digest(manifest),
              "registration": entries[0]["registration"],
              "role": manifest["role"], "analysis_source_sha256": source_hashes(),
              "runtime_source_match": source_hashes() == manifest["source_sha256"],
              "scheduled": len(rows), "finished": len(finished), "rows": rows,
              "expansion": (analysis.expansion(rows, len(manifest["cells"]), manifest["rubric"])
                            if manifest["role"] == "evaluation" else "NOT_SCORED")}
    if manifest["version"] != VERSION:  # a v1 report keeps exactly its v1 shape
        report["rubric"] = manifest["rubric"]
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "report"))
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--registration", default="PILOT")
    parser.add_argument("--max-cells", type=int, default=4)
    parser.add_argument("--clearance", help="operator note clearing a recorded provider stop")
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    root = args.data_root or data_root()
    if args.action == "run":
        run_campaign(manifest, root, args.registration, max_cells=args.max_cells,
                     clearance=args.clearance)
    else:
        print(json.dumps(report_campaign(manifest, root), indent=2))


if __name__ == "__main__":
    main()
