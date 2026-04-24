"""Confidence tracker - calculate confidence from score stability."""

from database.comparisons_table import get_effective_comparison_count
from shared.config import config

def calculate_confidence(
    filename: str,
    current_score: float = 0.5,
    comparison_count: int = 0,
    window_size: int = 100,
    days_window: int = 30,
) -> float:
    """
    Calculate confidence based on history index-based exponential decay.
    The newest comparison (index 0) has a 1.0 multiplier.
    The 10th newest comparison (index 9) has a 0.5 multiplier.
    """
    try:
        target_comparisons = int(config.get("ranking", {}).get("target_comparisons_for_max_confidence", 5))
    except Exception:
        target_comparisons = 5

    if target_comparisons <= 0:
        target_comparisons = 5

    # Compute the expected sum of weights for exactly `target_comparisons` direct comparisons
    target_divisor = sum(0.5 ** (i / 9.0) for i in range(target_comparisons))

    effective_count = 0.0
    if filename:
        effective_count = get_effective_comparison_count(filename)
    
    if effective_count <= 0 and comparison_count > 0:
        # Fallback if DB query fails: assume they were all direct comparisons recently
        effective_count = sum(0.5 ** (i / 9.0) for i in range(comparison_count))

    confidence = min(1.0, effective_count / target_divisor)
    return max(0.0, min(1.0, confidence))
