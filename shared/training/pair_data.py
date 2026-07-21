from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import numpy.typing as npt

from ..io import load_single_jsonl
from ..paths import comparisons_file
from tqdm import tqdm


def load_comparison_records() -> list[dict[str, Any]]:
    """Load the exported comparison edge list from disk."""

    records = load_single_jsonl(comparisons_file)
    return [dict(row) for row in records if isinstance(row, dict)]


def build_pairwise_dataset(
    x: npt.NDArray[np.float32],
    index_list: Sequence[str],
    comparisons: Sequence[dict[str, Any]],
) -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    list[int],
    npt.NDArray[np.float32],
    int,
]:
    """Expand comparison rows into ranker inputs.

    Each comparison becomes a 2-row query group: winner first, loser second.
    """

    index_lookup = {str(filename): idx for idx, filename in enumerate(index_list)}
    rows: list[npt.NDArray[np.float32]] = []
    labels: list[float] = []
    weights: list[float] = []
    groups: list[int] = []
    valid_pairs = 0
    with tqdm(
        total=len(comparisons), desc="Building pairwise dataset", unit=" pairs", delay=3.0
    ) as pbar:
        for comp in comparisons:
            pbar.update(1)
            filename_a = str(comp.get("filename_a", ""))
            filename_b = str(comp.get("filename_b", ""))
            winner = str(comp.get("winner", ""))

            if (
                filename_a not in index_lookup
                or filename_b not in index_lookup
                or winner not in index_lookup
            ):
                continue

            if winner == filename_a:
                winner_idx = index_lookup[filename_a]
                loser_idx = index_lookup[filename_b]
            elif winner == filename_b:
                winner_idx = index_lookup[filename_b]
                loser_idx = index_lookup[filename_a]
            else:
                continue

            weight_value = comp.get("weight", 1.0)
            weight = float(weight_value)
            rows.append(np.asarray(x[winner_idx], dtype=np.float32))
            rows.append(np.asarray(x[loser_idx], dtype=np.float32))
            labels.extend([1.0, 0.0])
            weights.extend([weight, weight])
            groups.append(2)
            valid_pairs += 1

    if rows:
        x_pair = np.asarray(rows, dtype=np.float32)
    else:
        raise ValueError("empty pairwise rows")

    y_pair = np.asarray(labels, dtype=np.float32)
    sample_weight = np.asarray(weights, dtype=np.float32)
    return x_pair, y_pair, groups, sample_weight, valid_pairs
