"""Workspace paths are data, never shell commands or executable code."""
import json
import os
import re
from itertools import islice
from pathlib import Path


def workspace_path(value, *, allow_root=False):
    location = os.environ.get("OPEN_LAB_WORKSPACE")
    if not location:
        raise ValueError("workspace is not configured")
    root = Path(location).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("workspace is not a directory")
    if allow_root and value == ".":
        return root, root
    parts = value.split("/")
    if (any(c in value for c in "\\:<>\"|?*\0")
            or any(ord(c) < 32 for c in value)
            or any(p in {"", ".", ".."} or p.endswith((".", " ")) for p in parts)):
        raise ValueError("use a relative workspace path with forward slashes")
    reserved = {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"}
    reserved.update(f"{prefix}{n}" for prefix in ("COM", "LPT") for n in range(1, 10))
    current = root
    for part in parts:
        if part.split(".")[0].upper() in reserved:
            raise ValueError("reserved file name")
        current = current / part
        if current.is_symlink() or current.is_junction():
            raise ValueError("links are not supported")
    if not current.resolve().is_relative_to(root):
        raise ValueError("path leaves workspace")
    return root, current



MAX_BYTES = 100_000
MAX_SKILLS = 1000


def skill_path(identifier):
    if not isinstance(identifier, str) or not re.fullmatch(r"[a-z][a-z0-9-]{0,63}", identifier):
        raise ValueError("skill id must be a lower-case slug of at most 64 characters")
    return workspace_path(f".skills/{identifier}.json")[1]


def store_files():
    _, store = workspace_path(".skills")
    if not store.exists():
        return []
    if not store.is_dir():
        raise ValueError("skill store is not a directory")
    # Bound inspection even if arbitrary workspace code has filled the directory.
    entries = list(islice(store.iterdir(), MAX_SKILLS + 1))
    if len(entries) > MAX_SKILLS:
        raise ValueError("skill store exceeds 1000 entries")
    if any(p.suffix != ".json" for p in entries):
        raise ValueError("unexpected skill store entry; inspect .skills")
    return sorted(entries)


def validate_skill(skill):
    if not isinstance(skill, dict) or set(skill) != {
            "id", "title", "summary", "steps", "examples", "tags", "provenance"}:
        raise ValueError("invalid stored skill fields")
    skill_path(skill["id"])
    for field, limit in (("title", 120), ("summary", 1000)):
        if not isinstance(skill[field], str) or not 1 <= len(skill[field]) <= limit:
            raise ValueError("invalid stored skill text")
    for field, count, length, minimum in (("steps", 50, 2000, 1),
                                         ("examples", 20, 4000, 1), ("tags", 20, 64, 0)):
        items = skill[field]
        if (not isinstance(items, list) or not minimum <= len(items) <= count
                or any(not isinstance(x, str) or not 1 <= len(x) <= length for x in items)):
            raise ValueError("invalid stored skill list")
    if skill["provenance"] != {"origin": "agent"}:
        raise ValueError("invalid stored skill provenance")


def read_skill(identifier):
    target = skill_path(identifier)
    if not target.exists():
        return None
    if not target.is_file() or target.stat().st_nlink != 1:
        raise ValueError("skill must be a regular file without links")
    with target.open("rb") as handle:
        data = handle.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError("stored skill exceeds 100000 bytes")
    try:
        skill = json.loads(data.decode("utf-8"))
    except (ValueError, UnicodeError):
        raise ValueError("invalid stored skill JSON") from None
    validate_skill(skill)
    if skill["id"] != identifier:
        raise ValueError("stored skill id differs from file name")
    return skill

def run(arguments):
    skill = read_skill(arguments["id"])
    return {"id": arguments["id"], "exists": skill is not None,
            **({"skill": skill} if skill is not None else {})}
