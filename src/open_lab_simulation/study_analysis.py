"""Deterministic descriptive analysis of verified exports, with cited assessments."""

from __future__ import annotations

import hashlib
import html
import itertools
import json
import re
import secrets
import shutil
from collections import Counter
from pathlib import Path

from .artifacts import read_index
from .contracts import Contracts, canonical_json, jsonl_lines
from .evidence import EvidenceError, _at, _json, _path, _plain, _snapshot, _validate, verify_export
from .policies import digest
from .world import file_sha256

SCHEMAS = ("common", "study-annotations", "study-analysis")
LIMITS = [
    "Descriptive records only; no autonomy score, new GO, causal matching or independence claim.",
    "Registration chronology, authenticity, effective compute and schedule completeness "
    "need external evidence.",
    "Recorded zero is not proof of absent uninstrumented intervention; "
    "unknown settings remain unknown.",
    "Annotation citations are checked; investigator interpretations "
    "are not independently adjudicated.",
]
HUMAN = [
    ["Effective provider identity/settings attestation and independent fresh-state assurance."],
    [
        "External preregistration chronology, intervention completeness "
        "and active/elapsed clock interpretation."
    ],
    ["Recorder deployment independence and externally trusted authenticity pins."],
    ["Original schedule completeness, effective seeds and independence of replicates."],
    ["Rubric adequacy, masking, reviewer agreement and interpretation quality."],
    ["Baseline currency for exact treatments and ADR-0017 change review."],
    ["Actual isolation/reliability exits and current owner resource/spending approvals."],
]


def _contracts(schemas):
    schemas = _plain(schemas)
    for name in SCHEMAS:
        doc = _json(_plain(schemas / "v1" / f"{name}.schema.json"))
        if doc.get("$id") != f"urn:open-lab:schema:{name}:v1":
            raise EvidenceError("unexpected analysis schema identity")
    return Contracts(schemas)


def _events(archive):
    return [
        json.loads(line)
        for line in jsonl_lines((archive / "events.jsonl").read_text(encoding="utf-8"))
    ]


def _trace(archive, config, record):
    events = _events(archive)
    seqs = {
        key: []
        for key in (
            "model_calls",
            "tool_calls",
            "tool_results_ok",
            "tool_results_failed",
            "agent_outputs",
            "preambles",
            "continuations",
            "demonstrations",
            "repairs",
            "context_changes",
            "marked_faults",
            "recovery",
            "checkpoints",
            "errors",
        )
    }
    calls = {}
    repairs = drops = blanks = 0
    for event in events:
        kind, p, seq = event["type"], event["payload"], event["seq"]
        ext = p.get("extensions", {})
        for event_type, key in (
            ("model.call", "model_calls"),
            ("tool.call", "tool_calls"),
            ("agent.output", "agent_outputs"),
            ("error", "errors"),
            ("recovery", "recovery"),
        ):
            if kind == event_type:
                seqs[key].append(seq)
        if kind == "model.call":
            repairs += p.get("repaired_tool_calls", 0)
            drops += p.get("context_dropped_messages", 0)
            blanks += p.get("context_truncated_tool_results", 0)
            if p.get("repaired_tool_calls", 0):
                seqs["repairs"].append(seq)
            if p.get("context_dropped_messages", 0) or p.get("context_truncated_tool_results", 0):
                seqs["context_changes"].append(seq)
        if kind == "tool.call":
            signature = digest([p["capability_id"], p["arguments"]])
            calls.setdefault(signature, []).append(seq)
            if p["capability_id"] == "capabilities.demonstrate":
                seqs["demonstrations"].append(seq)
        if kind == "tool.result":
            seqs["tool_results_ok" if p["status"] == "ok" else "tool_results_failed"].append(seq)
        if kind == "environment.message" and p["kind"] in {"preamble", "continuation"}:
            seqs["preambles" if p["kind"] == "preamble" else "continuations"].append(seq)
        if str(ext.get("stage", "")).startswith("benchmark_fault"):
            seqs["marked_faults"].append(seq)
        if kind.startswith("checkpoint."):
            seqs["checkpoints"].append(seq)
    return {
        "event_sha256": record["event_sha256"],
        "events_by_type": dict(sorted(Counter(e["type"] for e in events).items())),
        "seqs": seqs,
        "repaired_tool_calls": repairs,
        "context_dropped_messages": drops,
        "context_truncated_tool_results": blanks,
        "identical_call_repeats": [
            {"signature_sha256": key, "seqs": value}
            for key, value in sorted(calls.items())
            if len(value) > 1
        ],
        "provider_censored": record["final_state"] == "failed"
        and any("usage_limit_exceeded" in code for code in record["provider_terminal_codes"]),
        "no_recorded_tool_calls": not seqs["tool_calls"],
        "artifacts": read_index(archive),
        "model": record["model"],
        "policy": _json(archive / "policy.json") if (archive / "policy.json").is_file() else None,
        "requested": {
            k: config.get(k)
            for k in ("world", "orchestrator", "permissions", "limits", "continuation")
        }
        | {"prompt_sha256": digest(config["initial_prompt"])},
    }


def _flatten(value, prefix=""):
    if isinstance(value, dict) and value:
        result = {}
        for key, item in sorted(value.items()):
            result.update(_flatten(item, prefix + "/" + key.replace("~", "~0").replace("/", "~1")))
        return result
    return {prefix: canonical_json(value)}


def _groups(cells):
    groups = []
    for role, condition in sorted({(c["role"], c["condition"]) for c in cells}):
        rows = [c for c in cells if (c["role"], c["condition"]) == (role, condition)]
        groups.append(
            {
                "role": role,
                "condition": condition,
                "cells": [c["cell_id"] for c in rows],
                "planned": len(rows),
                "with_archive": sum(c["record"] is not None for c in rows),
                "no_archive": sum(c["record"] is None for c in rows),
                "declared_outcomes": dict(
                    sorted(Counter(c["declared_status"] for c in rows).items())
                ),
                "recorded_outcomes": dict(
                    sorted(Counter(c["record"]["final_state"] for c in rows if c["record"]).items())
                ),
                "provider_censored": sum(
                    bool(c["trace"] and c["trace"]["provider_censored"]) for c in rows
                ),
            }
        )
    return groups


def _contrasts(cells, configs):
    contrasts = []
    for role in sorted({c["role"] for c in cells}):
        conditions = sorted({c["condition"] for c in cells if c["role"] == role})
        for pair in itertools.combinations(conditions, 2):
            rows = [c for c in cells if c["role"] == role and c["condition"] in pair]
            flattened = [_flatten(configs[c["cell_id"]]) for c in rows]
            keys = sorted(set().union(*(set(f) for f in flattened)))
            differences = [k for k in keys if len({f.get(k) for f in flattened}) > 1]
            signatures = [
                digest({k: c["record"][k] for k in ("world_digest", "policy_sha256", "model")})
                for c in rows
                if c["record"]
            ]
            unknown = any(
                c["record"] is None
                or c["record"]["model"]["weight_digest"] is None
                or c["record"]["model"]["settings"] is None
                for c in rows
            )
            agreement = (
                "different"
                if len(set(signatures)) > 1
                else "unknown"
                if unknown
                else "equal_recorded_fields"
            )
            contrasts.append(
                {
                    "role": role,
                    "conditions": list(pair),
                    "cells": [c["cell_id"] for c in rows],
                    "configuration_differences": differences,
                    "recorded_field_agreement": agreement,
                    "causal_match": "not_established",
                    "independence": "not_established",
                }
            )
    return contrasts


def _empty_annotations(pin):
    return {
        "schema": "open-lab/study-annotations",
        "schema_version": "1.0.0",
        "export_sha256": pin,
        "rubric": {
            "id": "s010b-descriptive-v1",
            "definition": "Source-linked descriptive categories; no automatic autonomy score.",
        },
        "reviewer": {"id": "unreviewed", "masking": "unknown"},
        "limitations": [
            "No investigator annotations supplied; all behavioural assessments unreviewed."
        ],
        "entries": [],
    }


def _annotations(doc, contracts, pin, export, cells):
    _validate(contracts, doc, "study-annotations")
    if doc["export_sha256"] != pin:
        raise EvidenceError("annotations differ from the export pin")
    by_id = {c["cell_id"]: c for c in cells}
    seen, event_cache = set(), {}
    for entry in doc["entries"]:
        if entry["id"] in seen:
            raise EvidenceError("duplicate annotation id")
        seen.add(entry["id"])
        if entry["kind"] != "agent_claim" and (
            entry["quote"] is not None or entry["quote_pointer"] is not None
        ):
            raise EvidenceError("only agent claims may carry quotes")
        quoted = False
        for ref in entry["evidence"]:
            cell = by_id.get(ref["cell_id"])
            if cell is None or cell["trace"] is None:
                raise EvidenceError("annotation cell is unknown or has no recorded evidence")
            trace = cell["trace"]
            if ref["event_sha256"] != trace["event_sha256"]:
                raise EvidenceError("annotation event hash differs")
            archive = export / _path(cell["_archive_path"])
            if ref["cell_id"] not in event_cache:
                event_cache[ref["cell_id"]] = _events(archive)
            events = event_cache[ref["cell_id"]]
            if any(seq >= len(events) for seq in ref["seqs"]):
                raise EvidenceError("annotation sequence is outside its event log")
            for sha in ref["artifacts"]:
                if not any(
                    a["sha256"] == sha and a["observed_at_seq"] in ref["seqs"]
                    for a in trace["artifacts"]
                ):
                    raise EvidenceError("annotation artifact lacks a matching cited observation")
            if entry["kind"] == "agent_claim":
                for seq in ref["seqs"]:
                    event = events[seq]
                    if event["type"] not in {"agent.output", "agent.statement"}:
                        continue
                    if not isinstance(entry["quote_pointer"], str) or not entry["quote"]:
                        raise EvidenceError("agent quote needs a nonempty quote and pointer")
                    present, text = _at(event, entry["quote_pointer"])
                    quoted |= present and isinstance(text, str) and text == entry["quote"]
        if entry["kind"] == "agent_claim" and not quoted:
            raise EvidenceError("agent quote does not resolve exactly to a cited assertion")
    return doc


def analyze(export, schemas, annotations=None, expected_export_sha256=None):
    """Return a source-bound report without writing anywhere."""
    export = _plain(export)
    before = _snapshot(export)
    verified = verify_export(export, expected_export_sha256)
    contracts = _contracts(schemas)
    schema_pins = {n: file_sha256(Path(schemas) / "v1" / f"{n}.schema.json") for n in SCHEMAS}
    manifest = _json(export / "export.json")
    study = _json(export / _path(manifest["study_path"]))
    configs, cells = {}, []
    for planned, row in zip(study["cells"], manifest["cells"], strict=True):
        config = _json(
            export / _path(manifest["study_path"]).parent / _path(planned["configuration"]["path"])
        )
        configs[row["cell_id"]] = config
        cells.append(
            {
                **{
                    k: planned[k]
                    for k in ("cell_id", "role", "condition", "replicate", "seed_label")
                },
                **{k: row[k] for k in ("declared_status", "findings", "record")},
                "_archive_path": row["archive_path"],
                "trace": _trace(export / _path(row["archive_path"]), config, row["record"])
                if row["archive_path"]
                else None,
            }
        )
    doc = (
        annotations if annotations is not None else _empty_annotations(verified["manifest_sha256"])
    )
    doc = _annotations(doc, contracts, verified["manifest_sha256"], export, cells)
    for c in cells:
        del c["_archive_path"]
    identity_findings = any(
        c["findings"] or not c["record"] or c["record"]["model"]["weight_digest"] is None
        for c in cells
    )
    checks = [
        ["Configuration/initial-state inventory and recorded identity/settings retained."],
        [
            f"Registration kind: {study['registration']['kind']}; "
            "prompt/policy/intervention references retained."
        ],
        ["Export verified; event/artifact and annotation references checked."],
        [f"All {len(cells)} declared cells retained; roles grouped separately."],
        [f"{len(doc['entries'])} typed source-linked annotations; no composite score."],
        ["No automatic baseline-currency certification."],
        ["Offline analysis makes no model request and grants no launch authority."],
    ]
    report = {
        "schema": "open-lab/study-analysis",
        "schema_version": "1.0.0",
        "source": {
            "export_sha256": verified["manifest_sha256"],
            "study_sha256": manifest["study_sha256"],
            "analyzer_sha256": digest({
                p.name: hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                for p in sorted(Path(__file__).parent.glob("*.py"))
            }),
            "annotations_sha256": digest(doc),
            "schemas": schema_pins,
        },
        "study_id": study["study_id"],
        "question": study["question"],
        "registration": study["registration"],
        "comparison": study["comparison"],
        "requested_constants": manifest["constant_checks"],
        "cells": cells,
        "groups": _groups(cells),
        "contrasts": _contrasts(cells, configs),
        "annotations": doc,
        "validity_gates": [
            {
                "id": i + 1,
                "mechanical_status": "not_checked"
                if i >= 5
                else "findings"
                if (
                    i == 0
                    and identity_findings
                    or i == 1
                    and study["registration"]["kind"] != "prospective"
                    or i == 4
                    and not doc["entries"]
                )
                else "checked",
                "checks": checks[i],
                "human_or_unverified": HUMAN[i],
            }
            for i in range(7)
        ],
        "limitations": LIMITS + study["comparison"]["limitations"] + doc["limitations"],
    }
    _validate(contracts, report, "study-analysis")
    if _snapshot(export) != before or any(
        file_sha256(Path(schemas) / "v1" / f"{n}.schema.json") != pin
        for n, pin in schema_pins.items()
    ):
        raise EvidenceError("analysis source changed during computation")
    return report


def _bytes(doc):
    return (json.dumps(doc, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _display(value):
    # Operator and agent text is data, including in the Markdown rendering.
    text = html.escape(json.dumps(value, ensure_ascii=True))
    return re.sub(r"([\\`*_{}\[\]()#+.!|>\-])", r"\\\1", text)


def render(report):
    lines = [
        "# Descriptive study analysis",
        "",
        f"Study: `{report['study_id']}`",
        "",
        "All contrasts: causal matching and independence NOT ESTABLISHED.",
        "",
        "| Role | Condition | Planned | Archives | Missing | Provider censored |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for g in report["groups"]:
        role = _display(g["role"])
        condition = _display(g["condition"])
        lines.append(
            f"| {role} | {condition} | {g['planned']} | {g['with_archive']} | "
            f"{g['no_archive']} | {g['provider_censored']} |"
        )
    lines += ["", "| Cell | Declared | Final / reason | Tool calls | OK results | Revisions |",
              "| --- | --- | --- | ---: | ---: | ---: |"]
    for cell in report["cells"]:
        trace, record = cell["trace"], cell["record"]
        final = f"{record['final_state']} / {record['reason']}" if record else "unknown"
        counts = ([len(trace["seqs"]["tool_calls"]), len(trace["seqs"]["tool_results_ok"]),
                   len(trace["artifacts"])] if trace else ["unknown"] * 3)
        lines.append(f"| {_display(cell['cell_id'])} | {_display(cell['declared_status'])} | "
                     f"{_display(final)} | {counts[0]} | {counts[1]} | {counts[2]} |")
    lines += ["", "## Investigator annotations", ""]
    if not report["annotations"]["entries"]:
        lines.append("Unreviewed: no investigator annotations supplied.")
    for entry in report["annotations"]["entries"]:
        lines.append(f"- {_display(entry['kind'])} / {_display(entry['category'])} / "
                     f"{_display(entry['label'])}: {_display(entry['rationale'])}")
    lines += ["", "## Research gates", ""]
    for gate in report["validity_gates"]:
        lines.append(f"- Gate {gate['id']}: {gate['mechanical_status']}. "
                     f"Human/unverified: {_display(gate['human_or_unverified'])}")
    lines += [
        "",
        "Full source-linked observations, annotations and seven gate requirements: analysis.json.",
        "No scientific verdict is implied by record verification.",
        "",
    ]
    return "\n".join(lines).encode("utf-8")


def write_analysis(export, schemas, destination, annotation_path=None, expected_export_sha256=None):
    export, destination = _plain(export), _plain(destination)
    if destination.exists() or destination.is_relative_to(export):
        raise EvidenceError("analysis output must be new and outside the export")
    annotation_pin, doc = None, None
    if annotation_path is not None:
        annotation_path = _plain(annotation_path)
        if destination.is_relative_to(annotation_path.parent):
            raise EvidenceError("analysis output must be outside annotation inputs")
        annotation_pin = file_sha256(annotation_path)
        doc = _json(annotation_path)
    report = analyze(export, schemas, doc, expected_export_sha256)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.parent / f".{destination.name}.staging-{secrets.token_hex(8)}"
    stage.mkdir()
    for name in SCHEMAS:
        target = stage / "contracts/v1" / f"{name}.schema.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(schemas) / "v1" / f"{name}.schema.json", target)
    (stage / "annotations.json").write_bytes(_bytes(report["annotations"]))
    (stage / "analysis.json").write_bytes(_bytes(report))
    (stage / "report.md").write_bytes(render(report))
    result = verify_analysis(export, stage)
    if annotation_path is not None and file_sha256(annotation_path) != annotation_pin:
        raise EvidenceError("annotation input changed; staging retained")
    if destination.exists():
        raise EvidenceError("analysis destination appeared; staging retained")
    stage.rename(destination)
    return result


def verify_analysis(export, directory, expected_sha256=None):
    directory = _plain(directory)
    before = _snapshot(directory)
    expected_files = {"analysis.json", "annotations.json", "report.md"} | {
        f"contracts/v1/{n}.schema.json" for n in SCHEMAS
    }
    if set(before) != expected_files:
        raise EvidenceError("analysis file set differs")
    actual = file_sha256(directory / "analysis.json")
    if expected_sha256 is not None and actual != expected_sha256:
        raise EvidenceError("analysis differs from external pin")
    report = _json(directory / "analysis.json")
    _validate(_contracts(directory / "contracts"), report, "study-analysis")
    expected = analyze(
        export,
        directory / "contracts",
        _json(directory / "annotations.json"),
        report["source"]["export_sha256"],
    )
    if (
        report != expected
        or (directory / "analysis.json").read_bytes() != _bytes(expected)
        or (directory / "report.md").read_bytes() != render(expected)
    ):
        raise EvidenceError("analysis differs from reproducible source computation")
    if _snapshot(directory) != before:
        raise EvidenceError("analysis changed during verification")
    return {
        "verified": True,
        "analysis_sha256": actual,
        "cells": len(report["cells"]),
        "groups": len(report["groups"]),
        "annotations": len(report["annotations"]["entries"]),
    }
