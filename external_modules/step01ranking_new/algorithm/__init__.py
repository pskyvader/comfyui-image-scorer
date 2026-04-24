"""Algorithm module - ranking and comparison logic."""

from .merge_sort_ranker import (
    select_pair_for_comparison,
    update_scores_after_comparison,
)
from .transitive_inference import infer_score_from_chain

__all__ = [
    "select_pair_for_comparison",
    "update_scores_after_comparison",
    "infer_score_from_chain",
]
