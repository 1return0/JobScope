from __future__ import annotations

import math


def nearest_rank_percentile(
    values: list[float],
    percentile: float,
) -> float:
    if not values:
        raise ValueError("percentile values must not be empty")
    if not 0 < percentile <= 1:
        raise ValueError("percentile must be in the interval (0, 1]")
    ordered = sorted(values)
    rank = math.ceil(percentile * len(ordered))
    return ordered[rank - 1]
