"""Ordinary least squares with an intercept, on bounded paired numeric lists."""
from __future__ import annotations

import math
import statistics

MAX_ITEMS = 10000
MAX_ABS = 1e12


def _values(value, name):
    if not isinstance(value, list) or not 2 <= len(value) <= MAX_ITEMS:
        raise ValueError(f"{name} must contain 2 to 10000 numbers")
    if any(type(v) not in (int, float) or not -MAX_ABS <= v <= MAX_ABS for v in value):
        raise ValueError(f"{name} must contain finite numbers in [-1e12, 1e12], not booleans")
    return [float(v) for v in value]


def run(arguments: dict) -> dict:
    if not isinstance(arguments, dict) or set(arguments) != {"x", "y"}:
        raise ValueError("pass only the paired x and y lists")
    x, y = _values(arguments["x"], "x"), _values(arguments["y"], "y")
    if len(x) != len(y):
        raise ValueError("x and y must have equal lengths")
    x0, y0 = x[0], y[0]
    sx = max(abs(v - x0) for v in x)
    sy = max(abs(v - y0) for v in y)
    if sx == 0:
        raise ValueError("regression requires at least two distinct x values")
    if sy == 0:
        return {"count": len(x), "slope": 0.0, "intercept": y0, "r": None}
    # Affine normalization avoids squaring tiny spans or multiplying very small
    # variances in correlation. It preserves the OLS line and Pearson r.
    nx = [(v - x0) / sx for v in x]
    ny = [(v - y0) / sy for v in y]
    fit = statistics.linear_regression(nx, ny)
    slope = fit.slope * sy / sx
    if not math.isfinite(slope):
        raise ValueError("regression result cannot be represented as finite floating point")
    intercept = math.fsum((y0, fit.intercept * sy, -slope * x0))
    r = statistics.correlation(nx, ny)
    if not math.isfinite(intercept) or not math.isfinite(r):
        raise ValueError("regression result cannot be represented as finite floating point")
    return {"count": len(x), "slope": slope, "intercept": intercept, "r": r}
