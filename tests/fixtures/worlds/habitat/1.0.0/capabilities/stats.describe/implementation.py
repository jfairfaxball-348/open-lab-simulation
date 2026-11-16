"""Bounded descriptive statistics; no expressions, files or executable inputs."""
from __future__ import annotations

import math
import statistics

MAX_ITEMS = 10000
MAX_ABS = 1e12


def run(arguments: dict) -> dict:
    if not isinstance(arguments, dict) or set(arguments) != {"values"}:
        raise ValueError("pass only the values list")
    values = arguments["values"]
    if not isinstance(values, list) or not 1 <= len(values) <= MAX_ITEMS:
        raise ValueError("values must contain 1 to 10000 numbers")
    if any(type(v) not in (int, float) or not -MAX_ABS <= v <= MAX_ABS
           for v in values):
        raise ValueError("values must be finite numbers in [-1e12, 1e12]; booleans are not numbers")
    data = [float(v) for v in values]
    # Inclusive quartiles interpolate between observed endpoints; a singleton
    # distribution has the same value at every quantile.
    quartiles = statistics.quantiles(data, n=4, method="inclusive") if len(data) > 1 else data * 3
    result = {
        "count": len(data), "mean": statistics.fmean(data), "median": statistics.median(data),
        "variance": statistics.pvariance(data), "stdev": statistics.pstdev(data),
        "quantiles": dict(zip(("q1", "q2", "q3"), quartiles, strict=True)),
    }
    if not all(math.isfinite(result[k]) for k in ("mean", "median", "variance", "stdev")):
        raise ValueError("result cannot be represented as finite floating point")
    return result
