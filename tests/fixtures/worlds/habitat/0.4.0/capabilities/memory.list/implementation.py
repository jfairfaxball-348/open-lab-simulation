"""Agent-owned SQLite memory; no authority or infrastructure state is stored here."""
import os
import sqlite3
from contextlib import closing
from pathlib import Path


def run(arguments):
    location = os.environ.get("OPEN_LAB_WORKSPACE")
    if not location:
        raise ValueError("workspace is not configured")
    root = Path(location).resolve(strict=True)
    path = root / ".memory.sqlite"
    for name in (".memory.sqlite", ".memory.sqlite-journal", ".memory.sqlite-wal",
                 ".memory.sqlite-shm"):
        item = root / name
        if item.is_symlink() or item.is_junction():
            raise ValueError("memory links are not supported")
        if item.exists() and item.stat().st_nlink != 1:
            raise ValueError("memory hard links are not supported")
    if 'list' != "write" and not path.exists():
        return {"entries": []}
    try:
        # URI read mode prevents a read operation creating a database or journal.
        uri = path.as_uri() + ("?mode=rwc" if 'list' == "write" else "?mode=ro")
        with closing(sqlite3.connect(uri, uri=True, timeout=2)) as database, database:
            if 'list' == "write":
                database.execute("CREATE TABLE IF NOT EXISTS notes "
                                 "(key TEXT PRIMARY KEY, value TEXT NOT NULL, "
                                 "revision INTEGER NOT NULL)")
                database.execute("INSERT INTO notes VALUES (?, ?, 1) "
                                 "ON CONFLICT(key) DO UPDATE SET value=excluded.value, "
                                 "revision=notes.revision+1",
                                 (arguments["key"], arguments["value"]))
                revision = database.execute("SELECT revision FROM notes WHERE key=?",
                                            (arguments["key"],)).fetchone()[0]
                return {"key": arguments["key"], "revision": revision}
            limit = arguments.get("limit", 20)
            if 'list' == "search":
                rows = database.execute("SELECT key,value,revision FROM notes WHERE "
                                        "instr(lower(key),lower(?))>0 OR "
                                        "instr(lower(value),lower(?))>0 ORDER BY key LIMIT ?",
                                        (arguments["query"], arguments["query"], limit))
            else:
                rows = database.execute("SELECT key,value,revision FROM notes ORDER BY key LIMIT ?",
                                        (limit,))
            return {"entries": [{"key": k, "value": v, "revision": r} for k, v, r in rows]}
    except (sqlite3.Error, OSError):
        raise ValueError("memory database is unavailable or invalid") from None
