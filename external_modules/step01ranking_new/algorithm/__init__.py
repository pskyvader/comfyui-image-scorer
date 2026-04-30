"""Algorithm module - ranking and comparison logic."""

from .merge_sort_ranker import (
    select_pair_for_comparison,
    update_scores_after_comparison,
)

__all__ = [
    "select_pair_for_comparison",
    "update_scores_after_comparison",
]
