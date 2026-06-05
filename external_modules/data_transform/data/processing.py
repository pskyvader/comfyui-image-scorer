from __future__ import annotations

import time

import numpy as np

from shared.logger import get_logger

logger = get_logger(__name__)


def check_for_leakage(
    vectors_list: list[list[float]], scores_list: list[float]
) -> None:
    _start = time.perf_counter()
    if not vectors_list or not scores_list:
        raise RuntimeError(
            "check leakage: Empty vectors or scores list, "
            f"vectors list: {len(vectors_list)}, scores list: {len(scores_list)}"
        )
    if len(vectors_list) != len(scores_list):
        raise RuntimeError(
            "check leakage: lengths don't match, "
            f"vectors list: {len(vectors_list)}, scores list: {len(scores_list)}"
        )

    arr = np.array(vectors_list)
    y_arr = np.array(scores_list)
    corrs: list[float] = []
    for index in range(arr.shape[1]):
        column = arr[:, index]
        if np.allclose(column, column[0]):
            corrs.append(0.0)
            continue
        corr = np.corrcoef(column, y_arr)[0, 1]
        corrs.append(float(corr) if not np.isnan(corr) else 0.0)

    corrs_arr = np.array(corrs)
    leak_cols = np.where(np.abs(corrs_arr) >= 0.9999)[0].tolist()
    eq_cols = [
        index for index in range(arr.shape[1]) if np.allclose(arr[:, index], y_arr)
    ]
    leak_cols = sorted(set(leak_cols + eq_cols))
    if leak_cols:
        raise RuntimeError(
            "Detected feature columns that strongly match the target "
            f"(possible leakage): {leak_cols}. Fix the feature assembly to avoid "
            "including target values as features."
        )
