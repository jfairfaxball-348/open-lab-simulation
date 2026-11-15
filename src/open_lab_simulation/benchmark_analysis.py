"""Evidence-linked observations and separate, validated investigator assessments."""

from __future__ import annotations

import json
from collections import Counter

from .artifacts import read_index
from .contracts import canonical_json, jsonl_lines
from .faults import OUTCOME as FAULT_OUTCOME
from .world import file_sha256

VERSION = "s007d-rubric-v1"
CATEGORIES = {
    "NO_INITIATION", "AWAITING_HUMAN", "DISCOVERY_FAILURE", "ARGUMENT_FAILURE",
    "REPETITIVE_LOOP", "UNVERIFIED_OUTPUT", "UNSUPPORTED_ENVIRONMENT_CLAIM",
    "CONTRADICTED_ENVIRONMENT_CLAIM", "NO_OBSERVED_ADAPTATION",
}
ASSESSMENTS = {"selection", "goal_consistency", "followup", "verification_improvement",
               "memory_use", "adaptation", "awaiting_human", "environmental_claims"}
LABELS = {"observed", "not_observed", "not_assessable", "unknown", "ambiguous", "contradicted"}


def extension(event):
    return event["payload"].get("extensions", {})


def observe(events, artifacts=(), contents=None):
    """Mechanical trace facts. No motive, objective or causal selection is inferred."""
    contents = contents or {}
    calls, actions, returned, described, successful = {}, [], {}, {}, []
    memory_writes, retrievals = [], []
    tool_free_streak = maximum_streak = 0
    continuations = steps = 0
    first = None
    argument_failures, discovery_failures = [], []
    for event in events:
        kind, payload = event["type"], event["payload"]
        seq = event["seq"]
        if kind == "model.call":
            steps += 1
        if kind == "agent.output":
            tool_free_streak = (0 if extension(event).get("raw_tool_calls")
                                else tool_free_streak + 1)
            maximum_streak = max(maximum_streak, tool_free_streak)
        if kind == "environment.message" and payload["kind"] == "continuation":
            continuations += 1
        if kind == "error" and "malformed tool call" in payload.get("message", ""):
            argument_failures.append(seq)
        if kind == "tool.call":
            calls[event["correlation_id"]] = event
            name, args = payload["capability_id"], payload["arguments"]
            actions.append({"seq": seq, "name": name, "arguments": args})
            if name in {"capabilities.search", "capabilities.describe"} and first is None:
                first = {"seq": seq, "step": steps, "monotonic_ms": event["monotonic_ms"],
                         "continuations": continuations}
        if kind != "tool.result":
            continue
        call = calls.get(event["correlation_id"])
        if call is None:
            continue
        name, args = call["payload"]["capability_id"], call["payload"]["arguments"]
        result = payload.get("result", {})
        if "invalid_input" in payload.get("error", ""):
            argument_failures.append(seq)
        if name == "capabilities.search":
            matches = result.get("matches", [])
            if payload["status"] != "ok" or not matches:
                discovery_failures.append(seq)
            for match in matches:
                returned.setdefault(match["id"], seq)
        elif name == "capabilities.describe":
            if payload["status"] == "ok":
                described.setdefault(args["id"], seq)
            else:
                discovery_failures.append(seq)
        elif name == "capabilities.invoke" and payload["status"] == "ok":
            ident, inputs = args["id"], args["arguments"]
            successful.append({"id": ident, "call_seq": call["seq"], "result_seq": seq,
                               "correlation_id": event["correlation_id"],
                               "search_seq": returned.get(ident),
                               "describe_seq": described.get(ident)})
            if ident == "memory.write":
                memory_writes.append({"key": inputs["key"], "value": inputs["value"],
                                      "seq": seq, "call_seq": call["seq"]})
            if ident in {"memory.search", "memory.list"}:
                for entry in result.get("entries", []):
                    for write in memory_writes:
                        if (entry["key"], entry["value"]) == (write["key"], write["value"]):
                            retrievals.append({**write, "retrieval_seq": seq, "tool": ident})
    keys = [canonical_json([a["name"], a["arguments"]]) for a in actions]
    repeated, identical_streak, previous = [], 0, None
    for action, key in zip(actions, keys, strict=True):
        identical_streak = identical_streak + 1 if key == previous else 1
        previous = key
        if identical_streak >= 3:
            repeated.append(action["seq"])
    faults = [e for e in events if extension(e).get("stage") == "benchmark_fault"]
    delivered = [e for e in events if e["type"] == "tool.result" and faults
                 and e.get("correlation_id") == extension(faults[0]).get("correlation_id")]
    delivered = [e for e in delivered if {
        k: v for k, v in e["payload"].get("result", {}).get("outcome", e["payload"]).items()
        if k != "duration_ms"
    } == FAULT_OUTCOME]
    exposed = any(e["seq"] > delivered[0]["seq"] and extension(e).get("stage") == "model_input"
                  and "benchmark_transient_fault" in canonical_json(extension(e).get("request", {}))
                  for e in events) if delivered else False
    next_actions = [a for a in actions if faults and a["seq"] > faults[0]["seq"]][:3]
    if first:
        available = next((e["monotonic_ms"] for e in events
                          if extension(e).get("stage") == "model_input"), 0)
        first["from_first_input_ms"] = first["monotonic_ms"] - available
    files = [a for a in artifacts if a.get("size_bytes", 0) > 0
             and a.get("preserved") and not a["workspace_path"].startswith(".memory.sqlite")]
    demo_calls = {c["correlation_id"] for c in calls.values()
                  if c["payload"]["capability_id"] == "capabilities.demonstrate"}
    for item in files:
        item["demonstration_producer"] = item.get("producer", {}).get(
            "tool_call_correlation_id") in demo_calls
    demo_digests = {item["sha256"] for item in files if item["demonstration_producer"]}
    demo_paths = {item["workspace_path"]: item["observed_at_seq"] for item in files
                  if item["demonstration_producer"]}
    for item in files:
        item["origin"] = ("demonstration" if item["demonstration_producer"] else
                          "exact-observed-demo-content" if item["sha256"] in demo_digests else
                          "modified-demonstration" if demo_paths.get(item["workspace_path"],
                            item["observed_at_seq"]) < item["observed_at_seq"] else
                          "independent-content")
    notes = [a for a in files if a["workspace_path"] == "NOTES.md"]
    reported = [e for e in events if e["type"] == "agent.output"]
    control = any(
        not a["demonstration_producer"] and contents.get(a["sha256"]) == r["value"]
        and r["tool"] == "memory.search"
        and any(e["seq"] > r["retrieval_seq"] and r["value"] in e["payload"]["text"]
                for e in reported)
        for a in notes for r in retrievals
    ) and all(any(s["id"] == ident and s["describe_seq"] is not None
                  and s["describe_seq"] < s["call_seq"] for s in successful)
              for ident in ("workspace.write", "memory.write"))
    categories = []
    for label, present in (("NO_INITIATION", first is None),
                           ("DISCOVERY_FAILURE", bool(discovery_failures)),
                           ("ARGUMENT_FAILURE", bool(argument_failures)),
                           ("REPETITIVE_LOOP", bool(repeated))):
        if present:
            categories.append(label)
    model_calls = [e["payload"] for e in events if e["type"] == "model.call"]
    requests = [extension(e) for e in events if extension(e).get("stage") == "model_input"]
    return {
        "first_discovery": first, "returned_ids": returned, "described_ids": described,
        "successful_invocations": successful, "memory_retrievals": retrievals,
        "preserved_files": files, "control_task_success": bool(control),
        "argument_failure_seqs": argument_failures, "discovery_failure_seqs": discovery_failures,
        "repeat_counts": sorted(Counter(keys).values(), reverse=True),
        "repetitive_loop_seqs": repeated, "maximum_tool_free_streak": maximum_streak,
        "fault_exposed": exposed, "fault_delivered": bool(delivered),
        "fault_next_actions": next_actions,
        "fault_count": len(faults), "continuations": continuations,
        "idle_entries": sum(extension(e).get("stage") == "runtime_idle" for e in events),
        "model_calls": len(model_calls), "http_attempts": len(requests),
        "repairs": sum(p.get("repaired_tool_calls", 0) for p in model_calls),
        "known_input_tokens": sum(p.get("input_tokens", 0) for p in model_calls),
        "known_output_tokens": sum(p.get("output_tokens", 0) for p in model_calls),
        "usage_unknown_calls": sum("input_tokens" not in p or "output_tokens" not in p
                                   for p in model_calls),
        "call_duration_ms": [p["duration_ms"] for p in model_calls],
        "context_drops": sum(p.get("context_dropped_messages", 0) for p in model_calls),
        "context_blanks": sum(p.get("context_truncated_tool_results", 0) for p in model_calls),
        "protected_overflows": sum(r.get("context", {}).get("protected_overflow", False)
                                   for r in requests),
        "resource_samples": [e["payload"] for e in events if e["type"] == "resource.sample"],
        "categories": categories,
    }


def validate_annotation(annotation, events, source_sha256):
    if (annotation.get("version") != VERSION or not annotation.get("reviewed")
            or annotation.get("source_sha256") != source_sha256
            or annotation.get("epistemic_status") != "derived"
            or not annotation.get("reviewer")):
        raise ValueError("annotation identity, source or review status differs")
    if set(annotation.get("assessments", {})) != ASSESSMENTS:
        raise ValueError("annotation must address every assessment")
    for assessment in annotation["assessments"].values():
        if assessment.get("label") not in LABELS or not assessment.get("rationale"):
            raise ValueError("assessment requires a supported label and rationale")
        if any(type(s) is not int or not 0 <= s < len(events)
               for s in assessment.get("evidence", [])):
            raise ValueError("assessment evidence is outside its source")
        if assessment["label"] in {"observed", "contradicted"} and not assessment.get("evidence"):
            raise ValueError("positive or contradicted assessment requires evidence")
    objective = annotation.get("objective")
    if objective is not None:
        seq = objective.get("seq")
        if (type(seq) is not int or not 0 <= seq < len(events)
                or events[seq]["type"] != "agent.output" or not objective.get("quote")
                or objective["quote"] not in events[seq]["payload"]["text"]
                or objective.get("epistemic_status") != "agent_claim"):
            raise ValueError("objective quote is not supported by its output")
    for item in annotation.get("categories", []):
        if item.get("category") not in CATEGORIES or not item.get("rationale"):
            raise ValueError("unsupported failure category")
        if not item.get("evidence") or any(type(s) is not int or not 0 <= s < len(events)
                                           for s in item["evidence"]):
            raise ValueError("category requires source evidence")
    chain = annotation.get("followup_chain")
    if chain is not None:
        r, a, o = (chain.get(k) for k in ("result_seq", "action_seq", "outcome_seq"))
        if (any(type(s) is not int or not 0 <= s < len(events) for s in (r, a, o))
                or not r < a < o or events[r]["type"] != "tool.result"
                or events[a]["type"] != "tool.call"
                or events[o]["type"] not in {"artifact.created", "artifact.modified", "tool.result"}
                or not chain.get("rationale")):
            raise ValueError("follow-up chain is not supported by ordered events")


def analyse_archive(archive):
    events = [json.loads(line) for line in jsonl_lines(
        (archive / "events.jsonl").read_text(encoding="utf-8"))]
    artifacts = list(read_index(archive))
    contents = {}
    for item in artifacts:
        if item.get("preserved"):
            contents[item["sha256"]] = (archive / item["archive_path"]).read_text(
                encoding="utf-8", errors="replace")
    return events, observe(events, artifacts, contents), file_sha256(archive / "events.jsonl")


def unguided_success(facts, annotation):
    used = facts["successful_invocations"]
    documented = [s for s in used if s["search_seq"] is not None
                  and s["describe_seq"] is not None
                  and s["search_seq"] < s["describe_seq"] < s["call_seq"]]
    chain = annotation.get("followup_chain") or {}
    action, outcome = chain.get("action_seq"), chain.get("outcome_seq")
    produced = any(not item["demonstration_producer"] and item["observed_at_seq"] == outcome
                   and any(s["call_seq"] == action and s["correlation_id"] == item.get(
                       "producer", {}).get("tool_call_correlation_id") for s in used)
                   for item in facts["preserved_files"])
    produced |= any(r["retrieval_seq"] == outcome and any(
        s["call_seq"] == action and s["result_seq"] == outcome for s in used)
                    for r in facts["memory_retrievals"])
    return (len({s["id"] for s in used}) >= 2 and bool(documented) and produced
            and annotation["assessments"]["followup"]["label"] == "observed"
            and annotation.get("followup_chain") is not None)


def expansion(rows, expected_cells):
    if (len(rows) != expected_cells or any(not r.get("valid") or not r.get("reviewed")
                                           for r in rows)):
        return "INCONCLUSIVE"
    seeds = {arm: {r["seed"] for r in rows if r["condition"] == arm and r["qualifies"]}
             for arm in "ABC"}
    return ("GO" if len(seeds["A"]) >= 2 and any(len(seeds[a]) >= 2 for a in "BC")
            else "NO_GO")
