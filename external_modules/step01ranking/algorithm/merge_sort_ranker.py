"""Merge sort ranker — Slim orchestrator for O(N log N) pair selection.

This module is the public API surface. All pair-selection strategies live
in dedicated sub-modules; this file simply calls them in priority order
and re-exports every symbol that external callers (``ranking_api_v2``,
``__init__``) expect to find here.
"""

from typing import Any
from time import time
import logging

from shared.graph import crystal_graph

# ── Re-exports (keep the external API surface intact) ────────────────────
from .constants import (  # noqa: F401
    PAIR_TYPE_COLLAPSIBLE,
    PAIR_TYPE_WORST_WITH_WORST,
    PAIR_TYPE_UNCONNECTED,
    PAIR_TYPE_REFINEMENT,
    PAIR_TYPE_FALLBACK,
)
from .state import (  # noqa: F401
    get_last_pair_metadata,
    set_last_pair_metadata,
    get_cached_all_images,
)
from .graph_helpers import (  # noqa: F401
    is_collapsable_pair,
    filter_excluded_images,
    find_lowest_confidence_images,
)
from .comparison_recorder import (  # noqa: F401
    update_scores_after_comparison,
    record_comparison,
)

# ── Internal imports for the orchestrator ─────────────────────────────────
from .pair_orphan import find_orphan_pair
from .pair_extreme import find_extreme_pair_from_db
from .pair_low_count import find_low_count_pairs
from .pair_refinement import find_refinement_pairs
from external_modules.step01ranking.database.comparisons_table import (
    get_images_with_only_wins,
    get_images_with_only_losses,
)

logger: logging.Logger = logging.getLogger(__name__)


# ── Main entry point ─────────────────────────────────────────────────────


def select_pair_for_comparison(
    exclude_set: set[str] | None = None,
    exclude_chains: set[tuple[str, ...]] | None = None,
) -> tuple[str, str] | None:
    """Select the next pair of images to compare using crystal graph priorities.

    Strategy priority:
      0.  Orphan pairs (zero comparisons)
      0.5 Pure-winner / pure-loser pairs from the DB
      1.  Low comparison-count collapsible / merge pairs
      2.  Chain refinement (gap-based)
      3.  Score-range expansion
      4.  Loose pairs
      5.  Fallback: lowest-confidence cross-chain
    """
    start_timer = time()
    set_last_pair_metadata({})

    all_images: list[dict[str, Any]] = get_cached_all_images()

    if len(all_images) < 2:
        return None

    # Refresh crystal_graph if stale
    if crystal_graph.is_cache_stale():
        logger.debug(
            "[RANKER] Crystal graph cache is stale. Rebuilding from database..."
        )
        crystal_graph.rebuild_from_database(images=all_images)

        logger.debug(f"Crystal graph rebuild took {time() - start_timer:.2f} seconds")

    comp_count_lookup: dict[str, int] = {
        img["filename"]: img["comparison_count"] for img in all_images
    }
    score_lookup: dict[str, float] = {
        img["filename"]: img["score"] for img in all_images
    }
    logger.debug(
        f"[RANKER] Prepared lookup tables in {time() - start_timer:.2f} seconds"
    )

    candidate_images: list[dict[str, Any]] = filter_excluded_images(
        all_images,
        exclude_set,
    )

    logger.debug(
        f"[RANKER] {len(candidate_images)} candidate images after exclusion filtering in {time() - start_timer:.2f} seconds"
    )

    if len(candidate_images) < 2:
        return None

    candidate_filenames: set[str] = {img["filename"] for img in candidate_images}

    pair = None

    # Loop 0: Orphans
    # todo: condition: if no orphans exit immediately
    pair = find_orphan_pair(candidate_filenames, comp_count_lookup)

    logger.debug(
        f"[RANKER] Orphan pair selection took {time() - start_timer:.2f} seconds"
    )

    # Loop 1: Low comparison count pipeline
    only_wins: list[str] = [
        n for n in get_images_with_only_wins() if n in candidate_filenames
    ]
    only_losses: list[str] = [
        n for n in get_images_with_only_losses() if n in candidate_filenames
    ]

    if not pair:
        pair = find_low_count_pairs(
            candidate_filenames, comp_count_lookup, only_wins, only_losses
        )

        logger.debug(
            f"[RANKER] Low-count pair selection took {time() - start_timer:.2f} seconds"
        )

    # Loop 1.5: Pure winners/losers from DB
    if not pair:
        pair = find_extreme_pair_from_db(
            candidate_filenames, comp_count_lookup, only_wins, only_losses
        )

        logger.debug(
            f"[RANKER] Extreme pair selection took {time() - start_timer:.2f} seconds"
        )

    # Loop 2: Chain refinement
    if not pair and len(crystal_graph.get_all_chains()) > 1:
        pair: tuple[str, str] | None = find_refinement_pairs(
            candidate_filenames, comp_count_lookup, score_lookup
        )
        logger.debug(
            f"[RANKER] Total time taken for pair selection: {time() - start_timer:.2f} seconds"
        )
    return pair
