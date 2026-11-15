"""Offline study inventories and byte-preserving exports; never launches a run."""
from __future__ import annotations

import hashlib
import json
import re
import secrets
import shutil
from pathlib import Path, PurePosixPath

from . import __version__
from .archive import verify_archive
from .contracts import Contracts, jsonl_lines
from .paths import is_link
from .policies import digest
from .world import file_sha256

SCHEMAS = ("common", "experiment-config", "study-manifest", "evidence-export")
STOPPED = {"completed", "failed", "paused", "terminated"}
DEVICES = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
           *(f"LPT{i}" for i in range(1, 10))}


class EvidenceError(ValueError):
    pass


def _path(value):
    if (not isinstance(value, str) or not value or value.startswith("/")
            or any(c in value for c in '\\:<>"|?*') or any(ord(c) < 32 for c in value)):
        raise EvidenceError("unsafe relative evidence path")
    parts = value.split("/")
    if any(p in {"", ".", ".."} or p.endswith((".", " "))
           or p.split(".")[0].upper() in DEVICES for p in parts):
        raise EvidenceError("nonportable evidence path")
    return Path(*parts)


def _plain(path):
    path = Path(path).absolute()
    if any(is_link(p) for p in (path, *path.parents)):
        raise EvidenceError("evidence paths must not contain links or junctions")
    return path.resolve()


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _validate(contracts, value, name):
    errors = contracts.validate(value, name)
    if errors:
        raise EvidenceError(f"invalid {name}: {errors[0]}")


def _contracts(directory):
    directory = _plain(directory)
    for name in SCHEMAS:
        path = _plain(directory / "v1" / f"{name}.schema.json")
        doc = _json(path)
        if doc.get("$id") != f"urn:open-lab:schema:{name}:v1":
            raise EvidenceError("unexpected evidence/configuration schema identity")
    return Contracts(directory)


def _register(seen, relative):
    parts = _path(relative).parts
    for i in range(1, len(parts) + 1):
        prefix = "/".join(parts[:i])
        if seen.setdefault(prefix.casefold(), prefix) != prefix:
            raise EvidenceError("case-colliding evidence paths are not portable")


def _snapshot(directory):
    directory = _plain(directory)
    if not directory.is_dir():
        raise EvidenceError("missing archive or export directory")
    seen, files = {}, {}

    def visit(parent):
        for path in sorted(parent.iterdir()):
            _plain(path)
            relative = path.relative_to(directory).as_posix()
            _register(seen, relative)
            if path.is_dir():
                visit(path)
            elif path.is_file():
                files[relative] = {"sha256": file_sha256(path),
                                   "size_bytes": path.stat().st_size}
            else:
                raise EvidenceError("evidence contains a non-regular file")

    visit(directory)
    return files


def _omitted(relative):
    path = PurePosixPath(relative)
    if not path.parts:
        return False
    if len(path.parts) == 1 and path.name in {
        ".writer.guard", ".writer.lock", ".supervisor.json", "runtime.stderr.log",
    }:
        return True
    return (path.parts[0] == "segments" and
            ((path.name.startswith("writer-") and path.suffix == ".json")
             or path.name.endswith(".stderr.log")))


def _inputs(study_path, contracts):
    study = _json(study_path)
    _validate(contracts, study, "study-manifest")
    ids, archives, seen, files, configs = set(), set(), {}, {study_path.name: study_path}, {}
    _register(seen, study_path.name)
    for cell in study["cells"]:
        if cell["cell_id"] in ids:
            raise EvidenceError("duplicate study cell id")
        ids.add(cell["cell_id"])
        outcome = cell["outcome"]
        archive = outcome["archive"]
        if outcome["status"] == "recorded" and archive is None:
            raise EvidenceError("a recorded cell needs an archive identity")
        if outcome["status"] in {"not_started", "launch_failed"} and archive is not None:
            raise EvidenceError("an unstarted/failed-launch cell cannot claim an archive")
        if archive is not None:
            identity = (archive["experiment_id"], archive["run_id"])
            if identity in archives:
                raise EvidenceError("duplicate archive used by multiple cells")
            archives.add(identity)
        ref = cell["configuration"]
        _register(seen, ref["path"])
        path = _plain(study_path.parent / _path(ref["path"]))
        config = _json(path)
        _validate(contracts, config, "experiment-config")
        if digest(config) != ref["sha256"]:
            raise EvidenceError("study configuration differs from its canonical pin")
        if ref["path"] == study_path.name:
            raise EvidenceError("study and configuration paths collide")
        files[ref["path"]] = path
        configs[cell["cell_id"]] = config
    for ref in study["attachments"]:
        _register(seen, ref["path"])
        path = _plain(study_path.parent / _path(ref["path"]))
        if file_sha256(path) != ref["sha256"]:
            raise EvidenceError("study attachment differs from its byte pin")
        if ref["path"] == study_path.name or (
            ref["path"] in files and files[ref["path"]] != path
        ):
            raise EvidenceError("study attachment path collision")
        files[ref["path"]] = path
    return study, files, configs


def _at(value, pointer):
    if not pointer.startswith("/") or re.search(r"~(?![01])", pointer):
        raise EvidenceError("constant fields must be valid JSON pointers")
    for token in pointer[1:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(value, dict) and token in value:
            value = value[token]
        elif isinstance(value, list) and re.fullmatch(r"0|[1-9][0-9]*", token):
            if int(token) >= len(value):
                return False, None
            value = value[int(token)]
        else:
            return False, None
    return True, value


def _constants(study, configs):
    checks = []
    for role in sorted({c["role"] for c in study["cells"]}):
        ids = [c["cell_id"] for c in study["cells"] if c["role"] == role]
        for pointer in study["comparison"]["constant_fields"]:
            values = [_at(configs[ident], pointer) for ident in ids]
            status = ("missing" if not all(present for present, _ in values) else
                      "equal" if len({digest(v) for _, v in values}) == 1 else "different")
            checks.append({"role": role, "pointer": pointer, "status": status, "cells": ids})
    return checks


def _record(archive, expected_config):
    if (archive / ".writer.lock").exists():
        raise EvidenceError("archive has a writer marker; no live/stale snapshot export")
    summary = _json(archive / "summary.json")
    if summary.get("final_state") not in STOPPED:
        raise EvidenceError("only stopped archives with a summary can be exported")
    problems = verify_archive(archive)
    if problems:
        raise EvidenceError(f"archive verification failed: {problems[0]}")
    events = [json.loads(line) for line in jsonl_lines(
        (archive / "events.jsonl").read_text(encoding="utf-8"))]
    initial = events[0]["payload"].get("extensions", {}).get("initial_workspace")
    initial_state = "unknown"
    if (isinstance(initial, dict) and isinstance(initial.get("files"), list)
            and all(isinstance(p, str) for p in initial["files"])
            and type(initial.get("memory_present")) is bool
            and initial.get("sha256") == digest(initial["files"])):
        initial_state = "nonempty" if initial["files"] or initial["memory_present"] else "empty"
    model_path, policy_path = archive / "model.json", archive / "policy.json"
    model = _json(model_path) if model_path.is_file() else {}
    ledger = archive / "provider-ledger.jsonl"
    outcomes = []
    if ledger.is_file():
        raw = ledger.read_text(encoding="utf-8")
        if raw and not raw.endswith("\n"):
            raise EvidenceError("provider ledger has a torn final record")
        outcomes = [json.loads(line) for line in jsonl_lines(raw)]
    return {
        "event_sha256": file_sha256(archive / "events.jsonl"),
        "configuration_matches": digest(_json(archive / "experiment.json")) == digest(
            expected_config),
        "final_state": summary["final_state"], "reason": summary["reason"],
        "world_digest": _json(archive / "world-release.json")["content_digest"]["value"],
        "policy_sha256": digest(_json(policy_path)) if policy_path.is_file() else None,
        "model": {"model_id": model.get("model_id"), "provider_id": model.get("provider_id"),
                  "weight_digest": model.get("identity", {}).get("digest"),
                  "settings": model.get("settings")},
        "initial_state": initial_state, "provider_ledger_present": ledger.is_file(),
        "provider_terminal_codes": [str(e.get("code", "unknown")) for e in outcomes
                                    if e.get("kind") == "outcome"
                                    and e.get("result") == "terminal"],
    }


def _inventory(study, configs, root):
    rows = []
    for cell in study["cells"]:
        ref = cell["outcome"]["archive"]
        relative = (f"runs/{ref['experiment_id']}/{ref['run_id']}" if ref else None)
        record = _record(_plain(root / _path(relative)), configs[cell["cell_id"]]) if ref else None
        findings = []
        if record:
            if not record["configuration_matches"]:
                findings.append("CONFIGURATION_MISMATCH")
            if record["initial_state"] != "empty":
                findings.append(f"INITIAL_STATE_{record['initial_state'].upper()}")
        rows.append({"cell_id": cell["cell_id"], "role": cell["role"],
                     "declared_status": cell["outcome"]["status"], "archive_path": relative,
                     "record": record, "findings": findings})
    return rows


def _file_index(snapshot):
    return [{"path": path, **snapshot[path]} for path in sorted(snapshot)]


def verify_export(directory, expected_sha256=None):
    directory = _plain(directory)
    files = _snapshot(directory)
    manifest_path = directory / "export.json"
    actual_pin = file_sha256(manifest_path)
    if expected_sha256 is not None and actual_pin != expected_sha256:
        raise EvidenceError("export manifest differs from its external pin")
    contracts = _contracts(directory / "contracts")
    manifest = _json(manifest_path)
    _validate(contracts, manifest, "evidence-export")
    files.pop("export.json")
    if manifest["files"] != _file_index(files):
        raise EvidenceError("export file set, sizes or hashes differ")
    study_path = _path(manifest["study_path"])
    if study_path.parts[0] != "inputs":
        raise EvidenceError("study must be inside export inputs")
    if file_sha256(directory / study_path) != manifest["study_sha256"]:
        raise EvidenceError("study differs from its exported digest")
    study, _, configs = _inputs(directory / study_path, contracts)
    rows = _inventory(study, configs, directory)
    if manifest["cells"] != rows or manifest["constant_checks"] != _constants(study, configs):
        raise EvidenceError("export inventory differs from its records")
    prefixes = {row["archive_path"] for row in rows if row["archive_path"]}
    seen = set()
    for omission in manifest["omissions"]:
        path = _path(omission["path"])
        prefix = Path(*path.parts[:3]).as_posix()
        rest = Path(*path.parts[3:]).as_posix()
        if (prefix not in prefixes or not _omitted(rest) or omission["path"] in files
                or omission["path"] in seen):
            raise EvidenceError("invalid or duplicate operational omission")
        seen.add(omission["path"])
    return {"verified": True, "manifest_sha256": actual_pin,
            "external_pin_checked": expected_sha256 is not None, "cells": len(rows),
            "archives": len(prefixes), "inventory_findings": sum(len(r["findings"]) for r in rows)}


def export_study(study_path, schemas, root, destination):
    study_path, root, destination = map(_plain, (study_path, root, destination))
    if destination.exists():
        raise EvidenceError("export destination already exists; never overwritten")
    if destination.is_relative_to(root) and not destination.is_relative_to(root / "exports"):
        raise EvidenceError("inside the data root, exports may only be written under exports/")
    if destination.is_relative_to(study_path.parent):
        raise EvidenceError("export destination must be outside study inputs")
    contracts = _contracts(schemas)
    study_pin = file_sha256(study_path)
    study, inputs, configs = _inputs(study_path, contracts)
    input_pins = {source: file_sha256(source) for source in inputs.values()}
    if file_sha256(study_path) != study_pin:
        raise EvidenceError("study changed during validation")
    snapshots, sources, omissions = {}, {}, []
    for row in study["cells"]:
        ref = row["outcome"]["archive"]
        if not ref:
            continue
        relative = f"runs/{ref['experiment_id']}/{ref['run_id']}"
        archive = _plain(root / _path(relative))
        if (archive / ".writer.lock").exists():
            raise EvidenceError("archive has a writer marker; no live/stale snapshot export")
        before = _snapshot(archive)
        _record(archive, configs[row["cell_id"]])
        snapshots[archive] = before
        for path, info in before.items():
            if _omitted(path):
                omissions.append({"path": f"{relative}/{path}", **info})
            else:
                sources[f"{relative}/{path}"] = archive / _path(path)
    for path, source in inputs.items():
        sources[f"inputs/{path}"] = source
    for name in SCHEMAS:
        sources[f"contracts/v1/{name}.schema.json"] = Path(schemas) / "v1" / f"{name}.schema.json"
    seen = {}
    for path in sources:
        _register(seen, path)
    source_pins = {key: {"sha256": file_sha256(source), "size_bytes": source.stat().st_size}
                   for key, source in sources.items()}
    rows = _inventory(study, configs, root)
    # Stage only at an allowed new output parent. Failed stages remain evidence;
    # there is no cleanup path that could accidentally delete source material.
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.parent / f".{destination.name}.staging-{secrets.token_hex(8)}"
    stage.mkdir()
    for path, source in sorted(sources.items()):
        target = stage / _path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    for archive, before in snapshots.items():
        if (archive / ".writer.lock").exists() or _snapshot(archive) != before:
            raise EvidenceError("source archive changed during export; staging retained")
    for key, source in sources.items():
        if file_sha256(source) != source_pins[key]["sha256"]:
            raise EvidenceError("source input changed during export; staging retained")
    if any(file_sha256(path) != pin for path, pin in input_pins.items()):
        raise EvidenceError("study input changed after validation; staging retained")
    manifest = {
        "schema": "open-lab/evidence-export", "schema_version": "1.0.0",
        "study_path": f"inputs/{study_path.name}", "study_sha256": file_sha256(study_path),
        "exporter": {"id": "ols-evidence", "version": "1.0.0",
                     "source_sha256": hashlib.sha256(
                         Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()},
        "files": _file_index(source_pins), "omissions": sorted(omissions, key=lambda x: x["path"]),
        "cells": rows, "constant_checks": _constants(study, configs),
    }
    (stage / "export.json").write_text(json.dumps(manifest, sort_keys=True, indent=2,
                                                 ensure_ascii=False) + "\n",
                                       encoding="utf-8", newline="\n")
    result = verify_export(stage)
    if destination.exists():
        raise EvidenceError("export destination appeared during staging; not overwritten")
    stage.rename(destination)
    return {**result, "simulation_version": __version__}
