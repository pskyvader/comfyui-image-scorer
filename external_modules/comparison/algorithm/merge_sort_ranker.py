"""Public orchestration layer for step01 pair selection and comparison recording."""

from __future__ import annotations

from collections import deque
from typing import Any
from time import time
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

# Last 3 pairs' images kept in the exclude set so they cannot
# reappear as opponents in consecutive calls even if the server-
# side LRU has already cycled them out.
_last_pair_cooldown: deque[str] = deque(maxlen=6)


def select_pair_for_comparison(
    _start = time.perf_counter()
    _start = time.perf_counter()
    exclude_set: set[str] | None = None,
    logger.debug("select_pair_for_comparison took %.4fs", time.perf_counter() - _start)
) -> tuple[str, str] | None:
    """Select the next pair of images to compare."""

    started = time()
    set_last_pair_metadata({})
    all_images: list[dict[str, Any]] = get_cached_all_images()
    if len(all_images) < 2:
        return None

    if crystal_graph.is_cache_stale():
        crystal_graph.rebuild_from_database(images=all_images)

    # Build combined exclude set: caller LRU + cooldown
    combined_exclude: set[str] = set()
    if exclude_set:
        combined_exclude.update(exclude_set)
    for fn in list(_last_pair_cooldown):
        combined_exclude.add(fn)
    exclude_set = combined_exclude if combined_exclude else None

    original_count = len(all_images)
    candidate_images = filter_excluded_images(all_images, exclude_set)
    excluded_count = original_count - len(candidate_images)
    if excluded_count > 0:
        logger.info(
            "LRU excluded %d/%d images from selection pool",
            excluded_count,
            original_count,
        )

    if len(candidate_images) < 2:
        logger.warning(
            "select_pair_for_comparison: only %d images after exclusion, cannot form pair",
            len(candidate_images),
        )
        return None

    pair, meta = select_pair(all_images, candidate_images)
    if not pair:
        return None

    pair = pair[0], pair[1]
    a, b = pair
    _last_pair_cooldown.append(a)
    _last_pair_cooldown.append(b)

    meta = dict(meta)
    meta.setdefault("selection_time_ms", round((time() - started) * 1000.0, 2))
    set_last_pair_metadata(meta)
    logger.info(
        "SELECTED pair: %s vs %s (type=%s, time=%.1fms)",
        pair[0],
        pair[1],
        meta["pair_type"],
        meta["selection_time_ms"],
    )
    logger.debug(f"total time: {time() - started:.2f}s")
    return pair


def clear_pair_cooldown() -> None:
    _start = time.perf_counter()
    _start = time.perf_counter()
    _last_pair_cooldown.clear()
    logger.debug("clear_pair_cooldown took %.4fs", time.perf_counter() - _start)
