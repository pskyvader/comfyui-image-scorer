"""Public orchestration layer for step01 pair selection and comparison recording."""

from __future__ import annotations

from collections import deque
from typing import Any
import time
import logging

from shared.graph import crystal_graph

from .state import (
    set_last_pair_metadata,
    get_cached_all_images,
)
from .graph_helpers import (
    filter_excluded_images,
)
from .pair_active import select_pair

logger = logging.getLogger(__name__)


def select_pair_for_comparison(
    exclude_set: set[str] | None,
) -> tuple[str, str] | None:
    """Select the next pair of images to compare."""
    _start = time.perf_counter()
    started = time.perf_counter()
    set_last_pair_metadata({})
    all_images: list[dict[str, Any]] = get_cached_all_images()
    if len(all_images) < 2:
        logger.warning(
            f"select_pair_for_comparison: <2 images ({time.perf_counter() - _start:.4f}s)"
        )
        return None

    if crystal_graph.is_cache_stale():
        crystal_graph.rebuild_from_database()

    combined_exclude: set[str] = set()
    if exclude_set:
        combined_exclude.update(exclude_set)
    candidate_images = filter_excluded_images(all_images, combined_exclude)

    if len(candidate_images) < 2:
        logger.warning(
            "select_pair_for_comparison: only %d images after exclusion, cannot form pair",
            len(candidate_images),
        )
        return None

    pair, meta = select_pair(all_images, candidate_images)
    if not pair:
        logger.warning(
            f"select_pair_for_comparison: no pair ({time.perf_counter() - _start:.4f}s)"
        )
        return None

    pair = pair[0], pair[1]
    meta = dict(meta)
    meta["selection_time_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
    set_last_pair_metadata(meta)
    return pair
