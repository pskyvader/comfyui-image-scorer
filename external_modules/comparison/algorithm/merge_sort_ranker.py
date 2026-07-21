"""Public orchestration layer for step01 pair selection and comparison recording."""

from __future__ import annotations

from collections import deque
from typing import Any
import time

from ....shared.logger import get_logger, ModuleLogger
from ....shared.graph.crystal_graph import crystal_graph

from .state import (
    get_cached_all_images,
)
from .graph_helpers import (
    filter_excluded_images,
)
from .phase_order import select_pair

logger: ModuleLogger = get_logger(__name__)


def select_pair_for_comparison(
    exclude_set: set[str] | None,
) -> tuple[tuple[str, str] | None, int | None]:
    """Select the next pair of images to compare.

    Returns ``(pair, phase_index)`` where ``pair`` is ``(filename_a,
    filename_b)`` or ``None`` and ``phase_index`` is an int (0=seed,
    1=anchor, 2=collapsible, 3=chain_merge, 4=refine, 5=fallback) or ``None``.
    """
    _start = time.perf_counter()
    all_images: list[dict[str, Any]] = get_cached_all_images()
    if len(all_images) < 2:
        logger.warning(
            f"select_pair_for_comparison: <2 images ({time.perf_counter() - _start:.4f}s)"
        )
        return None, None

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
        return None, None

    pair, phase_index = select_pair(all_images, candidate_images)

    if not pair:
        logger.warning(f"select_pair_for_comparison: no pair", start_timer=_start)
        return None, None

    # logger.debug(
    #     f"select_pair_for_comparison: selected pair {pair} in phase {phase_index}",
    #     start_timer=_start,
    # )
    return pair, phase_index
