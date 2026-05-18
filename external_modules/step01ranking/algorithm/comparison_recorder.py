"""Comparison recording and score updates.

Handles the side-effects of a comparison: score delta calculation,
database writes, JSON history sync, and cache invalidation.
"""

from typing import Any
import math
import logging
from datetime import datetime, timezone

from database.images_table import (
    get_image as get_image_data,
    update_image_score,
    update_image_confidence,
)
from database.comparisons_table import add_comparison
from algorithm.confidence_tracker import calculate_confidence
from file_management.path_handler import append_comparison_history_to_json
from shared.graph import crystal_graph
from .state import invalidate_images_cache

logger = logging.getLogger(__name__)


def update_scores_after_comparison(
    winner_filename: str,
    loser_filename: str,
    winner_data: dict,
    loser_data: dict,
    impact_factor: float = 1.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Update scores for both images after a comparison."""
    winner_score = winner_data["score"]
    loser_score = loser_data["score"]
    winner_comp_count = winner_data["comparison_count"]
    loser_comp_count = loser_data["comparison_count"]

    # New "Fast Jump" formula: 0.5 at 0 comps, 0.05 at 10 comps
    winner_delta = 0.5 * math.exp(-0.23 * winner_comp_count) * impact_factor
    loser_delta = 0.5 * math.exp(-0.23 * loser_comp_count) * impact_factor

    new_winner_score = max(0.0, min(1.0, winner_score + winner_delta))
    new_loser_score = max(0.0, min(1.0, loser_score - loser_delta))

    winner_data["score"] = new_winner_score
    winner_data["comparison_count"] = winner_comp_count + 1
    loser_data["score"] = new_loser_score
    loser_data["comparison_count"] = loser_comp_count + 1

    return winner_data, loser_data


def record_comparison(
    filename_a: str,
    filename_b: str,
    winner: str,
    impact_factor: float = 1.0,
    transitive_depth: int = 0,
) -> bool:
    """Record a comparison and update image scores/confidence."""
    data_a = get_image_data(filename_a)
    data_b = get_image_data(filename_b)

    if not data_a or not data_b or filename_a == filename_b:
        return False

    if winner == filename_a:
        winner_data, loser_data = data_a, data_b
        winner_filename, loser_filename = filename_a, filename_b
    else:
        winner_data, loser_data = data_b, data_a
        winner_filename, loser_filename = filename_b, filename_a

    winner_data, loser_data = update_scores_after_comparison(
        winner_filename, loser_filename, winner_data, loser_data, impact_factor
    )

    ts = datetime.now(timezone.utc).isoformat()
    comp_id = add_comparison(
        filename_a,
        filename_b,
        winner,
        impact_factor,
        transitive_depth,
        timestamp=ts,
    )
    if not comp_id:
        logger.error(
            f"[RECORD] Failed to insert comparison into DB: {filename_a} vs {filename_b}, winner: {winner}"
        )
        return False


    update_image_score(winner_filename, winner_data["score"])
    update_image_confidence(
        winner_filename,
        calculate_confidence(
            winner_filename, winner_data["score"], winner_data["comparison_count"]
        ),
        winner_data["comparison_count"],
    )
    update_image_score(loser_filename, loser_data["score"])
    update_image_confidence(
        loser_filename,
        calculate_confidence(
            loser_filename, loser_data["score"], loser_data["comparison_count"]
        ),
        loser_data["comparison_count"],
    )


    entry_winner = {
        "comparison_id": comp_id,
        "other": loser_filename,
        "winner": True,
        "weight": impact_factor,
        "transitive_depth": transitive_depth,
        "timestamp": ts,
    }
    entry_loser = {
        "comparison_id": comp_id,
        "other": winner_filename,
        "winner": False,
        "weight": impact_factor,
        "transitive_depth": transitive_depth,
        "timestamp": ts,
    }

    winner_json_saved = append_comparison_history_to_json(
        winner_filename,
        entry_winner,
        new_score=winner_data["score"],
        new_confidence=calculate_confidence(
            winner_filename,
            winner_data["score"],
            winner_data["comparison_count"],
        ),
    )
    loser_json_saved = append_comparison_history_to_json(
        loser_filename,
        entry_loser,
        new_score=loser_data["score"],
        new_confidence=calculate_confidence(
            loser_filename,
            loser_data["score"],
            loser_data["comparison_count"],
        ),
    )

    if not winner_json_saved or not loser_json_saved:
        logger.error(
            f"[RECORD] History sync failed for comparison {comp_id} (winner: {winner_filename}, loser: {loser_filename})"
        )
        raise RuntimeError(
            f"Comparison history save incomplete for comp_id={comp_id} "
            f"{winner_filename}<->{loser_filename}: winner_saved={winner_json_saved}, loser_saved={loser_json_saved}"
        )
    logger.info(
        f"[RECORD] Successfully synced history to JSON for comparison {comp_id}"
    )

    # Incrementally update crystal_graph to avoid 5-second rebuild latency
    crystal_graph.apply_comparison(winner_filename, loser_filename)

    # Invalidate images cache so next call gets fresh comparison counts
    invalidate_images_cache()

    return True
