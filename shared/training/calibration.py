from __future__ import annotations

from typing import Any

import numpy as np


def _as_1d_float_array(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    return array[np.isfinite(array)]


def _strictly_increasing(values: np.ndarray) -> np.ndarray:
    adjusted = np.asarray(values, dtype=np.float32).reshape(-1).copy()
    if adjusted.size <= 1:
        return adjusted

    for i in range(1, adjusted.size):
        if not np.isfinite(adjusted[i]):
            adjusted[i] = adjusted[i - 1]
        if adjusted[i] <= adjusted[i - 1]:
            adjusted[i] = np.nextafter(adjusted[i - 1], np.float32(np.inf))
    return adjusted


def build_score_calibration(
    raw_scores: Any,
    target_scores: Any,
    num_points: int = 257,
) -> dict[str, Any] | None:
    """Build a monotonic quantile-based score calibration table."""

    raw = _as_1d_float_array(raw_scores)
    target = _as_1d_float_array(target_scores)
    if raw.size == 0 or target.size == 0:
        return None

    probabilities = np.linspace(0.0, 1.0, num_points, dtype=np.float32)
    raw_quantiles = np.quantile(raw, probabilities).astype(np.float32)
    target_quantiles = np.quantile(target, probabilities).astype(np.float32)

    raw_quantiles = _strictly_increasing(raw_quantiles)
    target_quantiles = np.maximum.accumulate(target_quantiles.astype(np.float32))

    return {
        "kind": "quantile_match",
        "probabilities": probabilities,
        "raw_quantiles": raw_quantiles,
        "target_quantiles": target_quantiles,
    }


def extract_score_calibration(data: dict[str, Any] | None) -> dict[str, Any] | None:
    if not data or "score_calibration" not in data:
        return None

    calibration = data["score_calibration"]
    if isinstance(calibration, np.ndarray) and calibration.shape == ():
        calibration = calibration.item()

    return calibration if isinstance(calibration, dict) else None


def apply_score_calibration(
    raw_scores: Any, calibration: dict[str, Any] | None
) -> np.ndarray:
    scores = np.asarray(raw_scores, dtype=np.float32).reshape(-1)
    if not calibration:
        return scores

    probabilities = np.asarray(calibration.get("probabilities"), dtype=np.float32).reshape(
        -1
    )
    raw_quantiles = np.asarray(calibration.get("raw_quantiles"), dtype=np.float32).reshape(
        -1
    )
    target_quantiles = np.asarray(
        calibration.get("target_quantiles"), dtype=np.float32
    ).reshape(-1)

    if (
        probabilities.size < 2
        or raw_quantiles.size != probabilities.size
        or target_quantiles.size != probabilities.size
    ):
        return scores

    percentiles = np.interp(scores, raw_quantiles, probabilities, left=0.0, right=1.0)
    calibrated = np.interp(
        percentiles,
        probabilities,
        target_quantiles,
        left=float(target_quantiles[0]),
        right=float(target_quantiles[-1]),
    )
    return np.asarray(calibrated, dtype=np.float32).reshape(-1)
