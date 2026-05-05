"""Merge sort ranker - O(N log N) pair selection and scoring."""

from typing import Any
from database.images_table import (
    get_all_images,
    get_image as get_image_data,
    update_image_score,
    update_image_confidence,
)
from database.comparisons_table import (
    add_comparison,
)
from algorithm.confidence_tracker import calculate_confidence
from file_management.path_handler import (
    append_comparison_history_to_json,
)
from shared.config import config
from shared.graph import crystal_graph
from datetime import datetime, timezone
import heapq
import math
import random
import time
from file_management.path_handler import (
    append_comparison_history_to_json,
)
from shared.config import config
from shared.graph import crystal_graph
from datetime import datetime, timezone
import heapq
import math
import random
import time
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)

_images_cache: dict[str, Any] = {"data": None, "timestamp": 0.0}
_IMAGES_CACHE_TTL = 5.5  # Cache all_images for 2 seconds

_MAX_PAIR_CANDIDATES = 100


def _get_cached_all_images() -> list[dict[str, Any]]:
    """Return cached all_images list, refreshing if stale."""
    global _images_cache
    now = time.time()
    if (
        _images_cache["data"] is not None
        and (now - _images_cache["timestamp"]) < _IMAGES_CACHE_TTL
    ):
        return _images_cache["data"]

    data = get_all_images()
    _images_cache = {"data": data, "timestamp": now}
    return data


def _is_better_path(
    candidate_edges: int,
    candidate_impact: float,
    current: tuple[int, float] | None,
) -> bool:
    if current is None:
        return True

    current_edges, current_impact = current
    return candidate_edges < current_edges or (
        candidate_edges == current_edges and candidate_impact > current_impact
    )


def is_collapsable_pair(filename_a: str, filename_b: str) -> bool:
    """Check if a pair is collapsible (both top or both bottom nodes at same height)."""
    if filename_a not in crystal_graph.images or filename_b not in crystal_graph.images:
        return False

    # Get heights
    height_a = crystal_graph.height.get(filename_a)
    height_b = crystal_graph.height.get(filename_b)

    # Must be at same height
    if height_a != height_b:
        return False

    # Check if both are top nodes (no better connections)
    both_top = (
        len(crystal_graph.better.get(filename_a, [])) == 0
        and len(crystal_graph.better.get(filename_b, [])) == 0
    )

    # Check if both are bottom nodes (no worse connections)
    both_bottom = (
        len(crystal_graph.worse.get(filename_a, [])) == 0
        and len(crystal_graph.worse.get(filename_b, [])) == 0
    )

    return both_top or both_bottom


def select_pair_for_comparison(
    exclude_set: set[str] | None = None,
) -> tuple[str, str] | None:
    """
    Select the next pair of images to compare using crystal graph priorities.

    Priority order:
    1. Shortest chains (lowest height)
    2. Among those, chains with multiple top/bottom nodes (collapsible pairs)
    3. Pairs from unconnected chains (using are_in_same_path)
    4. If nothing at current height, go up one level and repeat
    5. Final fallback: compare lowest confidence images from different chains
    """
    total_time = time.time()
    step_time = time.time()

    all_images = _get_cached_all_images()
    logger.debug(
        f"[NEXT-PAIR] all images: {len(all_images)},  time: total: {time.time() - total_time}, step: {time.time() - step_time}"
    )
    step_time = time.time()

    if len(all_images) < 2:
        return None

    candidate_images = _filter_excluded_images(all_images, exclude_set)
    logger.debug(
        f"[NEXT-PAIR] filter_excluded_images:{len(candidate_images)} time: total: {time.time() - total_time}, step: {time.time() - step_time}"
    )
    step_time = time.time()

    if len(candidate_images) < 2:
        return None

    # Refresh crystal_graph if stale
    if crystal_graph.is_cache_stale():
        crystal_graph.build_from_database(images=all_images)
    logger.debug(
        f"[NEXT-PAIR] crystal_graph built, max_height: {crystal_graph.get_max_height()}, time: total: {time.time() - total_time}, step: {time.time() - step_time}"
    )
    step_time = time.time()

    # Get candidate filenames for quick lookup
    candidate_filenames = {img["filename"] for img in candidate_images}

    # Priority 1 & 2: Start from lowest height (shortest chains)
    all_heights: list[int] = sorted(crystal_graph.nodes_by_height.keys())
    logger.debug(f"[NEXT-PAIR] All heights: {all_heights}")

    for height in all_heights:
        nodes_at_height: list[str] = [
            n
            for n in crystal_graph.get_images_by_height(height)
            if n in candidate_filenames
        ]
        logger.debug(f"[NEXT-PAIR] Nodes at height {height}: {len(nodes_at_height)}")

        if len(nodes_at_height) < 2:
            continue
        random.shuffle(nodes_at_height)

        # Priority 2: Check for collapsable pairs (multiple top/bottom nodes)
        pair = _find_collapsable_pair_at_height(nodes_at_height, height)
        if pair:
            logger.debug(
                f"[NEXT-PAIR] collapsable pair at height {height}: {pair}, time: total: {time.time() - total_time}"
            )
            return pair

        # Priority 3: Find pairs from unconnected chains
        pair = _find_unconnected_pair(nodes_at_height)
        if pair:
            logger.debug(
                f"[NEXT-PAIR] unconnected pair at height {height}: {pair}, time: total: {time.time() - total_time}"
            )
            return pair

    # Priority 5: Final fallback - lowest confidence images from different chains
    low_conf_images = _find_lowest_confidence_images(candidate_images)
    pair = _find_fallback_cross_chain(low_conf_images)
    logger.debug(
        f"[NEXT-PAIR] fallback cross-chain pair:{pair} time: total: {time.time() - total_time}"
    )
    return pair


def _filter_excluded_images(
    images: list[dict[str, Any]], exclude_set: set[str] | None
) -> list[dict[str, Any]]:
    """Step 2: Filter out recently shown images."""
    if not exclude_set or len(images) <= 5:
        return images

    max_exclude = int(len(images) * 0.75)
    safe_exclude = (
        set(list(exclude_set)[:max_exclude])
        if len(exclude_set) > max_exclude
        else exclude_set
    )
    return [img for img in images if img["filename"] not in safe_exclude]


def _get_comparison_graph(all_images: list[dict[str, Any]]) -> "CrystalGraph":
    """Step 3: Get comparison graph data needed for filtering. Uses the global crystal_graph instance."""
    if crystal_graph.is_cache_stale():
        crystal_graph.build_from_database(images=all_images, include_transitive=False)
    return crystal_graph


def _find_lowest_confidence_images(
    images: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Step 4: Find images for comparison.

    Random selection from ends + middle for score diversity, limited to _MAX_PAIR_CANDIDATES.
    """
    if not images:
        return []

    # filter by confidence - allow a small margin (0.05) to include images in the same tier
    confidence = min(img["confidence"] for img in images)
    images = [img for img in images if img["confidence"] <= confidence + 0.05]
    # Use a random tie-breaker for score sorting to avoid filename bias
    # and ensure stratified sampling (ends/middle) picks from different parts of the pool.
    random_tiebreakers = {img["filename"]: random.random() for img in images}
    sorted_by_score = sorted(
        images,
        key=lambda img: (float(img["score"]), random_tiebreakers[img["filename"]]),
    )

    n = len(sorted_by_score)
    if n <= _MAX_PAIR_CANDIDATES:
        return sorted_by_score

    candidates = list(sorted_by_score)
    section_size = math.floor(_MAX_PAIR_CANDIDATES / 4)

    # candidates= first section + middle section + last section
    candidates = (
        candidates[:section_size]
        + candidates[(n // 2 - section_size) : (n // 2 + section_size + 1)]
        + candidates[n - section_size :]
    )
    # make sure the candidates are not duplicated, based on filename
    unique_candidates: list[dict[str, Any]] = []
    for img in candidates:
        if img["filename"] not in [img["filename"] for img in unique_candidates]:
            unique_candidates.append(img)

    random.shuffle(unique_candidates)

    return unique_candidates[:_MAX_PAIR_CANDIDATES]


def _find_collapsable_pair_at_height(
    nodes_at_height: list[str], height: int
) -> tuple[str, str] | None:
    """Find collapsable pairs at a given height (multiple top/bottom nodes)."""
    top_nodes = [n for n in nodes_at_height if not crystal_graph.better[n]]
    bottom_nodes = [n for n in nodes_at_height if not crystal_graph.worse[n]]

    logger.debug(
        f"[COLLAPSABLE-PAIR] Top nodes: {len(top_nodes)}, Bottom nodes: {len(bottom_nodes)}"
    )

    random.shuffle(top_nodes)
    random.shuffle(bottom_nodes)

    first = bottom_nodes
    second = top_nodes

    if random.random() < 0.3:
        first = top_nodes
        second = bottom_nodes

    for group in [first, second]:
        if len(group) >= 2:
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    a, b = group[i], group[j]
                    if not crystal_graph.are_in_same_path(a, b):
                        return (a, b)

    return None


def _find_unconnected_pair(nodes_at_height: list[str]) -> tuple[str, str] | None:
    """Find a pair of nodes that are not in the same path."""
    n = len(nodes_at_height)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = nodes_at_height[i], nodes_at_height[j]
            if not crystal_graph.are_in_same_path(a, b):
                return (a, b)
    return None


def _find_fallback_cross_chain(images: list[dict[str, Any]]) -> tuple[str, str] | None:
    """Fallback: find pair from different chains (not in same path)."""
    if len(images) < 2:
        return None

    # Sort by confidence (lowest first)
    sorted_images = sorted(images, key=lambda img: img["confidence"])

    n = len(sorted_images)
    for i in range(n):
        for j in range(i + 1, n):
            a = sorted_images[i]["filename"]
            b = sorted_images[j]["filename"]
            if not crystal_graph.are_in_same_path(a, b):
                return (a, b)
    return None


def update_scores_after_comparison(
    winner_filename: str,
    loser_filename: str,
    winner_data: dict,
    loser_data: dict,
    impact_factor: float = 1.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Update scores for both images after a comparison."""
    winner_score = winner_data.get("score", 0.5)
    loser_score = loser_data.get("score", 0.5)
    winner_comp_count = winner_data.get("comparison_count", 0)
    loser_comp_count = loser_data.get("comparison_count", 0)

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
    logger.info(
        f"[RECORD] Inserted comparison ID {comp_id}: {filename_a} vs {filename_b}, winner: {winner}, impact factor: {impact_factor}"
    )

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
    logger.debug(
        f"[RECORD] Updated scores - Winner {winner_filename}: {winner_data['score']:.3f}, Loser {loser_filename}: {loser_data['score']:.3f}"
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

    # Invalidate crystal_graph cache after recording
    crystal_graph._built_at = None

    return True
