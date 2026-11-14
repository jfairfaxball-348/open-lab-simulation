"""text.stats: exact character, word and line counts."""

from __future__ import annotations


def run(arguments: dict) -> dict:
    text = arguments["text"]
    # Windows and old Mac line endings count as one break each, like "\n".
    normalised = text.replace("\r\n", "\n").replace("\r", "\n")
    if normalised == "":
        lines = 0
    else:
        lines = normalised.count("\n") + (0 if normalised.endswith("\n") else 1)
    return {"characters": len(text), "words": len(text.split()), "lines": lines}
