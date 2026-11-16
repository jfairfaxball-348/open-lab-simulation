"""Read a bounded cited range from the immutable world library."""

import hashlib
import json
from pathlib import Path


def run(arguments):
    root = Path(__file__).resolve().parents[2] / "library"
    catalog = json.loads((root / "index.json").read_text(encoding="utf-8"))
    source = next((s for s in catalog["sources"] if s["id"] == arguments["source_id"]), None)
    if source is None:
        raise ValueError("unknown library source id; use library.search to discover sources")
    name = source["path"]
    if Path(name).name != name or ":" in name or "\\" in name:
        raise ValueError("invalid library catalog path")
    raw = (root / name).read_bytes()
    if len(raw) > 256 * 1024 or hashlib.sha256(raw).hexdigest() != source["sha256"]:
        raise ValueError("library source integrity failure")
    lines = raw.decode("utf-8").splitlines()
    start = arguments.get("start_line", 1)
    selected = lines[start - 1:start - 1 + arguments.get("line_count", 80)]
    text = "\n".join(selected)
    truncated = len(text) > 16000
    if truncated:
        raise ValueError("range exceeds 16000 characters; request fewer lines")
    end = min(start + len(selected) - 1, len(lines))
    return {"source": source, "start_line": start, "end_line": end,
            "total_lines": len(lines), "text": text,
            "next_line": end + 1 if selected and end < len(lines) else None}
