"""Small statistics helpers: percentiles, median and range, and two interval/test helpers."""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def percentile(values: Iterable[float], q: float) -> float | None:
    """Percentile by linear interpolation between order statistics (numpy's default method).

    `q` is in [0, 100]. An empty input has no percentile and gives None.
    """
    data = sorted(values)
    if not data:
        return None
    if not 0 <= q <= 100:
        raise ValueError(f"q must be in [0, 100], got {q}")
    if len(data) == 1:
        return float(data[0])
    pos = (len(data) - 1) * q / 100
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(data[lo])
    return float(data[lo] + (data[hi] - data[lo]) * (pos - lo))


def summarize(values: Iterable[float]) -> dict:
    """Count, mean, min, max and p50/p95/p99 of a list of numbers (None fields when empty)."""
    data = list(values)
    if not data:
        return {"n": 0, "mean": None, "min": None, "max": None,
                "p50": None, "p95": None, "p99": None}
    return {
        "n": len(data),
        "mean": sum(data) / len(data),
        "min": float(min(data)),
        "max": float(max(data)),
        "p50": percentile(data, 50),
        "p95": percentile(data, 95),
        "p99": percentile(data, 99),
    }


def median_range(values: Iterable[float]) -> dict:
    """Median with min and max, the way every table in this repository reports repeated runs."""
    data = [v for v in values if v is not None]
    if not data:
        return {"n": 0, "median": None, "min": None, "max": None}
    return {"n": len(data), "median": percentile(data, 50), "min": float(min(data)),
            "max": float(max(data))}


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion; (0, 1) when there are no trials."""
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar p-value from the two discordant counts.

    only_a: items arm A got right and arm B got wrong; only_b: the reverse.
    """
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def fmt_range(summary: dict, digits: int = 1) -> str:
    """'median (min-max)' for a median_range() dict; '-' when there is no data."""
    if not summary or summary.get("median") is None:
        return "-"
    return (f"{summary['median']:.{digits}f} ({summary['min']:.{digits}f}-"
            f"{summary['max']:.{digits}f})")


def pooled(seqs: Sequence[Sequence[float]]) -> list[float]:
    """Flatten several per-run lists into one list for pooled percentiles."""
    return [v for seq in seqs for v in seq]
