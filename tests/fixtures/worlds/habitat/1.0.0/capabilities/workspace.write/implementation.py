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
        root, target = workspace_path(arguments["path"])
        data = arguments["text"].encode("utf-8")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.stat().st_nlink != 1:
            raise ValueError("hard links are not supported")
        mode = "wb" if arguments.get("overwrite", False) else "xb"
        with target.open(mode) as handle:
            handle.write(data)
        return {"path": target.relative_to(root).as_posix(), "bytes_written": len(data)}
    except OSError:
        raise ValueError("cannot write workspace file (exists, inaccessible or invalid)") from None
