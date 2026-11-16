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


MAX_FILE_BYTES = 5_000_000
MAX_TEXT_CHARS = 100_000


def run(arguments):
    root, target = workspace_path(arguments["path"])
    relative = target.relative_to(root).as_posix()
    if not target.exists():
        return {"path": relative, "exists": False}
    if not target.is_file():
        raise ValueError("path is a directory; use workspace.list")
    if target.stat().st_size > MAX_FILE_BYTES:
        raise ValueError(f"file is larger than {MAX_FILE_BYTES} bytes")
    try:
        text = target.read_bytes().decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError("file is not UTF-8 text") from None
    except OSError:
        raise ValueError("cannot read workspace file") from None
    lines = text.splitlines(keepends=True)
    start = arguments.get("start_line", 1)
    limit = arguments.get("max_lines", 400)
    chosen = lines[start - 1:start - 1 + limit]
    body = "".join(chosen)
    truncated = start - 1 + len(chosen) < len(lines)
    if len(body) > MAX_TEXT_CHARS:
        body, truncated = body[:MAX_TEXT_CHARS], True
    end = start - 1 + body.count("\n") + (0 if body.endswith("\n") or not body else 1)
    return {"path": relative, "exists": True, "text": body, "start_line": start,
            "end_line": max(end, start - 1), "total_lines": len(lines),
            "truncated": truncated}
