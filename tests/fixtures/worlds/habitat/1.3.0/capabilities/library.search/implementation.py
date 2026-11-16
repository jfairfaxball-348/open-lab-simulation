"""Small bounded keyword retrieval, with no network or writable index."""

import hashlib
import json
import re
from pathlib import Path


def run(arguments):
    root = Path(__file__).resolve().parents[2] / "library"
    catalog = json.loads((root / "index.json").read_text(encoding="utf-8"))
    terms = set(re.findall(r"\w+", arguments.get("query", "").casefold()))
    matches = []
    for source in catalog["sources"]:
        name = source["path"]
        if Path(name).name != name or ":" in name or "\\" in name:
            raise ValueError("invalid library catalog path")
        raw = (root / name).read_bytes()
        if len(raw) > 256 * 1024 or hashlib.sha256(raw).hexdigest() != source["sha256"]:
            raise ValueError("library source integrity failure")
        text = raw.decode("utf-8")
        title = (source["title"] + " " + source["id"]).casefold()
        score = sum(5 * (term in title) + (term in text.casefold()) for term in terms)
        if terms and not score:
            continue
        lines = text.splitlines()
        first = next((i for i, line in enumerate(lines)
                      if any(term in line.casefold() for term in terms)), 0)
        matches.append({"source": source, "score": score, "line": first + 1,
                        "snippet": "\n".join(lines[first:first + 3])[:600]})
    matches.sort(key=lambda item: (-item["score"], item["source"]["id"]))
    return {"matches": matches[:arguments.get("limit", 5)], "total_matches": len(matches)}
