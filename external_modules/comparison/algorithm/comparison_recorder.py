"""Comparison recording and rating updates."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
import logging

from external_modules.database_structure.comparisons_table import add_comparison, comparison_exists_for_pair
from external_modules.database_structure.images_table import (
    get_image as get_image_data,
    update_image_rating_state,
)
from external_modules.database_structure.path_handler import sync_image_metadata_to_json
from shared.graph import crystal_graph
from .state import invalidate_images_cache
from .trueskill_rating import (
    Rating,
    public_score_from_rating,
    rating_from_row,
    update_ratings,
)

logger = logging.getLogger(__name__)


def update_scores_after_comparison(
    _start = time.perf_counter()
    _start = time.perf_counter()
    winner_filename: str,
    loser_filename: str,
    winner_data: dict[str, Any],
    loser_data: dict[str, Any],
    impact_factor: float = 1.0,
    logger.debug("update_scores_after_comparison took %.4fs", time.perf_counter() - _start)
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compatibility helper that now performs a TrueSkill update."""

    winner_rating, loser_rating = update_ratings(
        rating_from_row(winner_data), rating_from_row(loser_data)
    )
    winner_data = dict(winner_data)
    loser_data = dict(loser_data)
    winner_data["rating_mu"] = winner_rating.mu
    winner_data["rating_sigma"] = winner_rating.sigma
    winner_data["score"] = public_score_from_rating(winner_rating)
    winner_data["comparison_count"] = int(winner_data["comparison_count"]) + 1

    loser_data["rating_mu"] = loser_rating.mu
    loser_data["rating_sigma"] = loser_rating.sigma
    loser_data["score"] = public_score_from_rating(loser_rating)
    loser_data["comparison_count"] = int(loser_data["comparison_count"]) + 1
    return winner_data, loser_data


def _persist_image_state(str, data: dict[str, Any]) -> bool:
    result = update_image_rating_state(
    logger.debug("_persist_image_state took %.4fs", time.perf_counter() - _start)
    return result
        filename=filename,
        score=float(data["score"]),
        rating_mu=float(data["rating_mu"]),
        rating_sigma=float(data["rating_sigma"]),
        comparison_count=int(data["comparison_count"]),
        touch_timestamp=True,
    )


def record_comparison(
    _start = time.perf_counter()
    _start = time.perf_counter()
    filename_a: str,
    filename_b: str,
    winner: str,
    impact_factor: float = 1.0,
    transitive_depth: int = 0,
    logger.debug("record_comparison took %.4fs", time.perf_counter() - _start)
) -> bool:
    """Record one direct comparison and update both image ratings."""

    if comparison_exists_for_pair(filename_a, filename_b):
        logger.info("Skipping duplicate pair comparison for %s vs %s", filename_a, filename_b)
        return False

    data_a = get_image_data(filename_a)
    data_b = get_image_data(filename_b)
    if not data_a or not data_b or filename_a == filename_b:
        return False

    if winner == filename_a:
        winner_filename, loser_filename = filename_a, filename_b
        winner_data, loser_data = data_a, data_b
    else:
        winner_filename, loser_filename = filename_b, filename_a
        winner_data, loser_data = data_b, data_a

    winner_data, loser_data = update_scores_after_comparison(
        winner_filename, loser_filename, winner_data, loser_data, impact_factor
    )

    ts = datetime.now(timezone.utc).isoformat()
    comp_id = add_comparison(
        filename_a=filename_a,
        filename_b=filename_b,
        winner=winner,
        weight=impact_factor,
        transitive_depth=transitive_depth,
        timestamp=ts,
    )
    if not comp_id:
        logger.error(
            "Failed to insert comparison into DB: %s vs %s, winner=%s",
            filename_a,
            filename_b,
            winner,
        )
        return False

    if not _persist_image_state(winner_filename, winner_data):
        return False
    if not _persist_image_state(loser_filename, loser_data):
        return False

    from external_modules.database_structure.comparisons_table import get_all_comparisons

    all_comparisons = get_all_comparisons()
    saved_winner = sync_image_metadata_to_json(
        filename=winner_filename,
        score=float(winner_data["score"]),
        rating_mu=float(winner_data["rating_mu"]),
        rating_sigma=float(winner_data["rating_sigma"]),
        comparison_count=int(winner_data["comparison_count"]),
        all_comparisons=all_comparisons,
    )
    saved_loser = sync_image_metadata_to_json(
        filename=loser_filename,
        score=float(loser_data["score"]),
        rating_mu=float(loser_data["rating_mu"]),
        rating_sigma=float(loser_data["rating_sigma"]),
        comparison_count=int(loser_data["comparison_count"]),
        all_comparisons=all_comparisons,
    )
    if not saved_winner or not saved_loser:
        logger.error(
            "Failed to sync JSON history for comparison %s (%s vs %s)",
            comp_id,
            winner_filename,
            loser_filename,
        )
        return False

    crystal_graph.apply_comparison(winner_filename, loser_filename)
    invalidate_images_cache()
    return True
