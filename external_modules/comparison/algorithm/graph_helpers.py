"""Reusable graph-query helpers for the ranking algorithm.

Every helper operates on the global ``crystal_graph`` instance and avoids
duplicating the same node-grouping / filtering patterns that appear across
multiple pair-selection strategies.
"""

from typing import Any
from collections import defaultdict
import math
import random
import logging

from shared.graph import crystal_graph
from .constants import MAX_PAIR_CANDIDATES
import time

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Node topology queries
# ---------------------------------------------------------------------------

def is_top_node(str) -> bool:
    """Return True if *node* has no 'better than' links (i.e. is a top node)."""
    result = not crystal_graph._chain._better_than[node]
    logger.debug("is_top_node took %.4fs", time.perf_counter() - _start)
    return result


def is_bottom_node(str) -> bool:
    """Return True if *node* has no 'worse than' links (i.e. is a bottom node)."""
    result = not crystal_graph._chain._worse_than[node]
    logger.debug("is_bottom_node took %.4fs", time.perf_counter() - _start)
    return result


def get_node_component(str) -> int | None:
    """Return the component id of *node*, or ``None``."""
    result = crystal_graph._chain._node_component[node]
    logger.debug("get_node_component took %.4fs", time.perf_counter() - _start)
    return result


def get_component_members(int) -> list[str]:
    """Return the member list for a component id."""
    result = crystal_graph._chain._component_members[comp_id]
    logger.debug("get_component_members took %.4fs", time.perf_counter() - _start)
    return result


def get_chain_length(str) -> int:
    """Return the chain-length metric for *node*."""
    result = crystal_graph._chain._chain_length[node]
    logger.debug("get_chain_length took %.4fs", time.perf_counter() - _start)
    return result


# ---------------------------------------------------------------------------
# Grouping helpers
# ---------------------------------------------------------------------------

def group_nodes_by_extreme(
    _start = time.perf_counter()
    _start = time.perf_counter()
    nodes: list[str],
    logger.debug("group_nodes_by_extreme took %.4fs", time.perf_counter() - _start)
) -> tuple[dict[int, list[str]], dict[int, list[str]]]:
    """Group *nodes* into top-by-component and bottom-by-component dicts.

    Returns ``(top_by_component, bottom_by_component)`` where keys are
    component ids and values are lists of node filenames.
    """
    top_by_component: dict[int, list[str]] = defaultdict(list)
    bottom_by_component: dict[int, list[str]] = defaultdict(list)

    for n in nodes:
        comp_id = get_node_component(n)
        if comp_id is None:
            continue
        if is_top_node(n):
            top_by_component[comp_id].append(n)
        if is_bottom_node(n):
            bottom_by_component[comp_id].append(n)

    return top_by_component, bottom_by_component


# ---------------------------------------------------------------------------
# Public-API helpers (exposed via merge_sort_ranker re-exports)
# ---------------------------------------------------------------------------

def is_collapsable_pair(str, filename_b: str) -> bool:
    """Check if a pair is collapsible (both top or both bottom in same component, no common chains)."""
    node_a = crystal_graph.get_node(filename_a)
    node_b = crystal_graph.get_node(filename_b)
    if not node_a or not node_b:
        result = False
        logger.debug("is_collapsable_pair took %.4fs", time.perf_counter() - _start)
        return result

    # Must be in the same component
    comp_a = crystal_graph.get_component(node_id=filename_a)
    comp_b = crystal_graph.get_component(node_id=filename_b)
    if not comp_a or not comp_b or comp_a.id != comp_b.id:
        result = False
        logger.debug("is_collapsable_pair took %.4fs", time.perf_counter() - _start)
        return result

    # Check if both are top nodes (no better connections)
    both_top = node_a.is_top() and node_b.is_top()

    # Check if both are bottom nodes (no worse connections)
    both_bottom = node_a.is_bottom() and node_b.is_bottom()

    if not (both_top or both_bottom):
        result = False
        logger.debug("is_collapsable_pair took %.4fs", time.perf_counter() - _start)
        return result

    # Must not share any chains (not in same directed path)
    result = not crystal_graph.are_in_same_path(filename_a, filename_b)
    logger.debug("is_collapsable_pair took %.4fs", time.perf_counter() - _start)
    return result


# ---------------------------------------------------------------------------
# Filtering helpers
# ---------------------------------------------------------------------------

def filter_excluded_images(
    _start = time.perf_counter()
    _start = time.perf_counter()
    images: list[dict[str, Any]],
    exclude_set: set[str] | None,
    logger.debug("filter_excluded_images took %.4fs", time.perf_counter() - _start)
) -> list[dict[str, Any]]:
    """Remove images whose filename is in *exclude_set*."""
    if not exclude_set:
        return images

    safe_exclude = set(exclude_set)
    result = []
    for img in images:
        filename = img["filename"]
        if filename not in safe_exclude:
            result.append(img)

    logger.debug(
        "filter_excluded_images: removed %d of %d (exclude_set=%d)",
        len(images) - len(result), len(images), len(safe_exclude),
    )
    return result


def find_lowest_confidence_images(
    _start = time.perf_counter()
    _start = time.perf_counter()
    images: list[dict[str, Any]],
    logger.debug("find_lowest_confidence_images took %.4fs", time.perf_counter() - _start)
) -> list[dict[str, Any]]:
    """Select a diverse subset of high-uncertainty images for fallback pairing.

    Random selection from ends + middle for score diversity, limited to
    ``MAX_PAIR_CANDIDATES``.
    """
    if not images:
        return []

    sigma = max(float(img["rating_sigma"]) for img in images)
    images = [
        img for img in images if float(img["rating_sigma"]) >= sigma - 0.05
    ]
    # Use a random tie-breaker for score sorting to avoid filename bias
    # and ensure stratified sampling (ends/middle) picks from different parts of the pool.
    random_tiebreakers = {img["filename"]: random.random() for img in images}
    sorted_by_score = sorted(
        images,
        key=lambda img: (float(img["score"]), random_tiebreakers[img["filename"]]),
    )

    n = len(sorted_by_score)
    if n <= MAX_PAIR_CANDIDATES:
        return sorted_by_score

    candidates = list(sorted_by_score)
    section_size = math.floor(MAX_PAIR_CANDIDATES / 4)

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

    return unique_candidates[:MAX_PAIR_CANDIDATES]
