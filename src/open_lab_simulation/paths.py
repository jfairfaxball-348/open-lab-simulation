"""Where the shared data lives. Mirrors the rule in the Builder's CONTRACTS.md."""

from __future__ import annotations

import os
import stat
from pathlib import Path

_REPARSE_POINT = 0x400  # FILE_ATTRIBUTE_REPARSE_POINT


def data_root() -> Path:
    """``$OPEN_LAB_DATA``, or ``open-lab-data`` in the home directory."""
    configured = os.environ.get("OPEN_LAB_DATA")
    return Path(configured) if configured else Path.home() / "open-lab-data"


def is_link(path: Path) -> bool:
    """Whether a path is a link of any kind; host code never follows one.

    Besides symbolic links and junctions this includes every other Windows reparse
    point. A link made inside a WSL container appears on Windows as a WSL reparse
    point that ``Path.is_symlink()`` does not recognise and that cannot be opened.
    """
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return False
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & _REPARSE_POINT
    )


def walk_files(root: Path, on_link=None):
    """Yield regular files beneath ``root`` without entering or yielding any link.

    ``on_link`` is called with each link found; it may raise to stop the walk.
    """
    for directory, subdirectories, names in os.walk(root):
        kept = []
        for name in subdirectories:
            path = Path(directory, name)
            if is_link(path):
                if on_link:
                    on_link(path)
            else:
                kept.append(name)
        subdirectories[:] = kept
        for name in names:
            path = Path(directory, name)
            if is_link(path):
                if on_link:
                    on_link(path)
            elif path.is_file():
                yield path


def minimal_environment(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for child processes: nothing inherited that could carry a credential.

    Only the variables an interpreter needs to start on each platform are kept.
    """
    keep = ("SYSTEMROOT", "WINDIR", "SYSTEMDRIVE", "LANG", "LC_ALL")
    environment = {name: os.environ[name] for name in keep if name in os.environ}
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    environment.update(extra or {})
    return environment
