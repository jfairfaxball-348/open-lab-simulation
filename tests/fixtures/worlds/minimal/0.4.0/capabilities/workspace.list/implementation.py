"""Workspace paths are data, never shell commands or executable code."""
import os
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


def run(arguments):
    try:
        root, target = workspace_path(arguments.get("path", "."), allow_root=True)
        entries = []
        for child in target.iterdir():
            if len(entries) >= 1000:
                raise ValueError("directory exceeds 1000 entries")
            if child.is_symlink() or child.is_junction():
                raise ValueError("links are not supported")
            entries.append({"path": child.relative_to(root).as_posix(),
                            "kind": "directory" if child.is_dir() else "file"})
        return {"entries": sorted(entries, key=lambda e: e["path"])}
    except OSError:
        raise ValueError("cannot list workspace directory (missing or inaccessible)") from None
