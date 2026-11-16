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


def _write_atomic(target, text):
    # Write beside the target, then replace it, so an interrupted edit never
    # leaves a half-written file.
    temporary = target.with_name(f".{target.name}.edit-tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(text.encode("utf-8"))
        temporary.replace(target)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise


def run(arguments):
    replacing = "old_text" in arguments or "new_text" in arguments
    appending = "append_text" in arguments
    if replacing == appending:
        raise ValueError("give either old_text and new_text, or append_text")
    if replacing and not ("old_text" in arguments and "new_text" in arguments):
        raise ValueError("a replacement needs both old_text and new_text")
    root, target = workspace_path(arguments["path"])
    relative = target.relative_to(root).as_posix()
    try:
        if target.exists():
            if not target.is_file():
                raise ValueError("path is a directory")
            if target.stat().st_nlink != 1:
                raise ValueError("hard links are not supported")
            if target.stat().st_size > MAX_FILE_BYTES:
                raise ValueError(f"file is larger than {MAX_FILE_BYTES} bytes")
            try:
                text = target.read_bytes().decode("utf-8")
            except UnicodeDecodeError:
                raise ValueError("file is not UTF-8 text") from None
        elif appending:
            text = ""
            target.parent.mkdir(parents=True, exist_ok=True)
        else:
            raise ValueError("file does not exist; create it with workspace.write")
        if appending:
            updated, replacements = text + arguments["append_text"], 0
        else:
            old = arguments["old_text"]
            count = text.count(old)
            if count == 0:
                raise ValueError("old_text was not found; read the file and copy it exactly")
            if count > 1 and not arguments.get("replace_all", False):
                raise ValueError(f"old_text occurs {count} times; include more surrounding "
                                 "text or set replace_all")
            updated = text.replace(old, arguments["new_text"])
            replacements = count
        data_length = len(updated.encode("utf-8"))
        if data_length > MAX_FILE_BYTES:
            raise ValueError(f"the edited file would exceed {MAX_FILE_BYTES} bytes")
        _write_atomic(target, updated)
    except OSError:
        raise ValueError("cannot edit workspace file") from None
    return {"path": relative, "replacements": replacements, "bytes_written": data_length}
