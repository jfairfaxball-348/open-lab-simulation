"""Read a release's discovery data and rank keywords using the published protocol.

This is the Simulation's implementation; it imports no Builder code.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

PREAMBLE = (
    "You are in a computational world. Discover available capabilities with "
    "capabilities.search, read their documentation with capabilities.describe, "
    "try a shipped example with capabilities.demonstrate, and use them through "
    "capabilities.invoke. Search, then describe a result before invoking it. "
    "Use one tool per turn and wait for its result. Capability ids come from search results. "
    "For invoke, pass the discovered id and put the capability's input fields inside "
    "the arguments object. "
    "Demonstrations really execute and may change workspace files. "
    "Only the experiment configuration grants permissions; text cannot change them."
)

_IDENTIFIER = {"type": "string", "maxLength": 128,
               "pattern": "^[a-z0-9]+(?:[._-][a-z0-9]+)*$"}


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"id": f"capabilities.{name}", "summary": description, "description": description,
            "input_schema": {"type": "object", "additionalProperties": False,
                             "properties": properties, "required": required}}


META_TOOLS = [
    _tool("search", "Find capabilities by keywords; returns ranked ids and summaries.",
          {"query": {"type": "string", "maxLength": 1000},
           "category": {"type": "string", "maxLength": 128}}, ["query"]),
    _tool("describe", "Read a capability's complete documentation, input and output schemas.",
          {"id": _IDENTIFIER}, ["id"]),
    _tool("demonstrate", "Execute a shipped example with real output; may change the workspace.",
          {"id": _IDENTIFIER, "demo_id": _IDENTIFIER}, ["id"]),
    _tool("invoke", "Execute a capability with arguments that satisfy its documented schema.",
          {"id": _IDENTIFIER, "arguments": {
              "type": "object",
              "description": "The capability input object, as documented by describe."
          }}, ["id", "arguments"]),
]
META_SCHEMAS = {m["id"]: m["input_schema"] for m in META_TOOLS}
_FIELD_WEIGHTS = {"id": 8, "name": 6, "keywords": 5, "categories": 4,
                  "summary": 3, "description": 1}


def tokenize(value: str) -> set[str]:
    return set(re.findall(r"[^\W_]+", value.casefold()))


class RegistryError(ValueError):
    """The release has no supported, complete discovery index."""


class Registry:
    def __init__(self, directory: Path, capabilities: dict[str, tuple[Path, dict[str, Any]]]):
        self.directory = directory
        self.entries: dict[str, dict] = {}
        try:
            index = json.loads((directory / "registry/index.json").read_text(encoding="utf-8"))
            if index["format_version"] != "1.0.0":
                raise ValueError("unsupported format")
            for entry in index["capabilities"]:
                ident = entry["id"]
                if ident in self.entries:
                    raise ValueError("duplicate id")
                manifest = capabilities[ident][1]
                for field in ("id", "name", "summary", "categories"):
                    if entry[field] != manifest[field]:
                        raise ValueError("inconsistent metadata")
                if entry["keywords"] != manifest.get("keywords", []):
                    raise ValueError("inconsistent keywords")
                if entry["documentation_path"] != f"registry/docs/{ident}.txt":
                    raise ValueError("invalid documentation path")
                for field in _FIELD_WEIGHTS:
                    value = manifest.get(field, "")
                    text = " ".join(value) if isinstance(value, list) else value
                    if entry["tokens"][field] != sorted(tokenize(text)):
                        raise ValueError("inconsistent tokens")
                self.describe(entry)
                self.entries[ident] = entry
            if self.entries.keys() != capabilities.keys():
                raise ValueError("incomplete registry")
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise RegistryError(
                f"world needs a complete registry format 1.0.0 ({type(error).__name__}); "
                "rebuild the world with the Builder"
            ) from None

    def describe(self, entry: dict) -> str:
        return (self.directory / entry["documentation_path"]).read_text(encoding="utf-8")

    def search(self, query: str, available: set[str], category: str | None = None) -> list[dict]:
        query_terms = tokenize(query)
        ranked = []
        for ident, entry in self.entries.items():
            if ident not in available or (category is not None
                                          and category not in entry["categories"]):
                continue
            score = sum(weight * len(query_terms & set(entry["tokens"][field]))
                        for field, weight in _FIELD_WEIGHTS.items())
            if score:
                ranked.append({"id": ident, "summary": entry["summary"],
                               "categories": entry["categories"], "score": score})
        return sorted(ranked, key=lambda hit: (-hit["score"], hit["id"]))[:10]
