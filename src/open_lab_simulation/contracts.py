"""Validate documents against the contract schemas bundled in a world release.

The Simulation owns no schemas. Every release carries the exact versions it
was built with under ``contracts/``, and every run archive carries a copy, so
there is never a second opinion about what a valid document is. The rules
implemented here are specified in the Builder's ``docs/architecture/CONTRACTS.md``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from referencing import Registry, Resource

SCHEMA_PREFIX = "open-lab/"


def canonical_json(document: Any) -> str:
    """The one serialisation used for hashing and for lines of the event log."""
    return json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Contracts:
    """The schemas in one ``contracts/`` directory (major version 1)."""

    def __init__(self, contracts_dir: Path):
        self.directory = contracts_dir
        self.schemas: dict[str, dict[str, Any]] = {}
        registry = Registry()
        for path in sorted((contracts_dir / "v1").glob("*.schema.json")):
            schema = json.loads(path.read_text(encoding="utf-8"))
            self.schemas[path.name.removesuffix(".schema.json")] = schema
            registry = registry.with_resource(schema["$id"], Resource.from_contents(schema))
        if not self.schemas:
            raise FileNotFoundError("no contract schemas found")
        self._validators = {
            name: Draft202012Validator(schema, registry=registry)
            for name, schema in self.schemas.items()
        }

    def validate(self, document: Any, expected: str | None = None) -> list[str]:
        """Problems with a document; empty means valid. ``expected`` pins the contract name."""
        if not isinstance(document, dict):
            return ["document must be a JSON object"]
        schema_field = document.get("schema")
        if not isinstance(schema_field, str) or not schema_field.startswith(SCHEMA_PREFIX):
            return [f"/schema: must be a string starting with '{SCHEMA_PREFIX}'"]
        name = schema_field.removeprefix(SCHEMA_PREFIX)
        if expected is not None and name != expected:
            return [f"/schema: expected '{SCHEMA_PREFIX}{expected}'"]
        if name not in self.schemas or name == "common":
            return [f"/schema: unknown contract '{name}'"]
        try:
            major, minor, _ = (int(part) for part in str(document.get("schema_version")).split("."))
        except ValueError:
            return ["/schema_version: must be a MAJOR.MINOR.PATCH string"]
        if major != 1:
            return [f"/schema_version: unsupported major version {major}"]
        problems = [
            f"{''.join(f'/{part}' for part in error.absolute_path) or '<root>'}: {error.message}"
            for error in sorted(
                self._validators[name].iter_errors(document),
                key=lambda e: [str(part) for part in e.absolute_path],
            )
        ]
        # A newer minor version may use fields these schemas do not know.
        known_minor = int(self.schemas[name]["x-schema-version"].split(".")[1])
        if minor > known_minor:
            problems.append("/schema_version: newer than the bundled schema")
        return problems

    def version(self, name: str) -> str:
        return self.schemas[name]["x-schema-version"]


def validate_event_stream(contracts: Contracts, lines: list[str]) -> list[str]:
    """Check a whole event log: every event, the order, the run and the hash chain."""
    problems: list[str] = []
    first: dict[str, Any] | None = None
    previous_line: str | None = None
    for index, line in enumerate(lines):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            problems.append(f"line {index}: not JSON")
            previous_line = line
            continue
        problems.extend(
            f"line {index}{problem}" if problem.startswith("/") else f"line {index}: {problem}"
            for problem in contracts.validate(event, "observation-event")
        )
        if isinstance(event, dict):
            first = first or event
            if event.get("seq") != index:
                problems.append(f"line {index}: expected seq {index}, found {event.get('seq')!r}")
            for key in ("experiment_id", "run_id"):
                if event.get(key) != first.get(key):
                    problems.append(f"line {index}: {key} differs from the first event")
            expected = None if previous_line is None else sha256_text(previous_line)
            if event.get("prev_event_sha256") != expected:
                problems.append(f"line {index}: hash chain broken")
            if line != canonical_json(event):
                problems.append(f"line {index}: not in canonical form")
        previous_line = line
    return problems
