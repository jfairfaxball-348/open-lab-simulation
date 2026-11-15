"""Run the four frozen, owner-authorized S008c capability-check cells once.

Usage: uv run python scripts/s008c_check.py
Starts the next cell only; rerun explicitly for the next one. A started cell is
never repeated. This is a capability check, not the 24-cell baseline harness.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from open_lab_simulation.archive import verify_archive
from open_lab_simulation.benchmark import source_hashes
from open_lab_simulation.contracts import jsonl_lines
from open_lab_simulation.ownership import archive_lock
from open_lab_simulation.paths import data_root
from open_lab_simulation.policies import digest
from open_lab_simulation.supervisor import run_experiment
from open_lab_simulation.world import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quota-clearance", help="owner-confirmed plan availability after S007e")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    manifest = json.loads((repo / "campaigns/s008c-001.json").read_text(encoding="utf-8"))
    if manifest["source_sha256"] != source_hashes():
        raise SystemExit("Runtime source differs from the frozen check")
    if manifest["runner_sha256"] != file_sha256(Path(__file__)):
        raise SystemExit("Runner differs from the frozen check")
    if len(manifest["cells"]) != 4:
        raise SystemExit("Exactly four frozen cells are required")
    directory = data_root() / "capability-checks/s008c-001"
    directory.mkdir(parents=True, exist_ok=True)
    with archive_lock(directory, recover_stale=True):
        ledger = directory / "progress.json"
        progress = json.loads(ledger.read_text(encoding="utf-8")) if ledger.exists() else {
            "manifest_sha256": digest(manifest), "started": [], "outcomes": []}

        def save():
            temporary = ledger.with_suffix(".tmp")
            with temporary.open("w", encoding="utf-8", newline="\n") as stream:
                json.dump(progress, stream, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, ledger)

        if progress["manifest_sha256"] != digest(manifest):
            raise SystemExit("Manifest differs from the durable progress record")
        if len(progress["started"]) != len(progress["outcomes"]):
            raise SystemExit("An earlier cell is interrupted; no automatic replacement")
        if any(o["stop"] for o in progress["outcomes"]):
            raise SystemExit("A failed/censored cell stopped the check; owner review required")
        if not progress.get("quota_clearance"):
            if not args.quota_clearance or not args.quota_clearance.strip():
                raise SystemExit("Owner quota confirmation required; no run started")
            progress["quota_clearance"] = args.quota_clearance.strip()
            save()
        index = len(progress["started"])
        if index >= 4:
            raise SystemExit("All four authorized cells already started; no additional run")
        cell = manifest["cells"][index]
        path = repo / cell["config"]
        config = json.loads(path.read_text(encoding="utf-8"))
        if digest(config) != cell["config_sha256"] or config["limits"]["max_steps"] != 32:
            raise SystemExit("Cell differs from its frozen configuration")
        progress["started"].append(cell["id"])
        save()
        # A launch refusal also remains a started cell; it is never retried here.
        result = run_experiment(path)
        problems = verify_archive(result.archive)
        provider = [json.loads(line) for line in jsonl_lines(
            (result.archive / "provider-ledger.jsonl").read_text(encoding="utf-8"))]
        terminal = any(e.get("kind") == "outcome" and e.get("result") == "terminal"
                       for e in provider)
        outcome = {"cell": cell["id"], "run_id": result.run_id, "state": result.state,
                   "reason": result.reason, "archive_verify": problems,
                   "event_sha256": file_sha256(result.archive / "events.jsonl"),
                   "provider_sha256": file_sha256(result.archive / "provider-ledger.jsonl"),
                   "stop": bool(problems or terminal or result.state == "failed")}
        progress["outcomes"].append(outcome)
        save()
        print(json.dumps(outcome, indent=2))


if __name__ == "__main__":
    main()
