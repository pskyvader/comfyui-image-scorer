"""Analysis helpers - utility functions for analysis endpoints."""

from typing import Any
import logging
import time

logger = logging.getLogger(__name__)


def distribute(values: list[float], buckets: list[tuple[str, float]]) -> dict[str, int]:
    """Distribute values into named buckets by threshold."""
    result: dict[str, int] = {}
    for label, _ in buckets:
        result[label] = 0
    for v in values:
        for label, threshold in buckets:
            if v < threshold:
                result[label] += 1
                break
        else:
            result[buckets[-1][0]] += 1
    return result
