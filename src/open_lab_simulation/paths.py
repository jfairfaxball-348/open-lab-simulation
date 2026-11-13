"""Where the shared data lives. Mirrors the rule in the Builder's CONTRACTS.md."""

from __future__ import annotations

import os
from pathlib import Path


def data_root() -> Path:
    """``$OPEN_LAB_DATA``, or ``open-lab-data`` in the home directory."""
    configured = os.environ.get("OPEN_LAB_DATA")
    return Path(configured) if configured else Path.home() / "open-lab-data"


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
