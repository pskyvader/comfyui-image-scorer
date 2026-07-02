"""Reusable graph-query helpers for the ranking algorithm.

Every helper operates on the global ``crystal_graph`` instance and avoids
duplicating the same node-grouping / filtering patterns that appear across
multiple pair-selection strategies.
"""

from typing import Any
from collections import defaultdict
import math
import random
import time

from ....shared.graph.crystal_graph import crystal_graph
from .constants import MAX_PAIR_CANDIDATES

from ....shared.logger import get_logger, ModuleLogger
logger: ModuleLogger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Node topology queries
# ---------------------------------------------------------------------------


def is_top_node(node: str) -> bool:
    """Return True if node has no better-than links (i.e. is a top node)."""
    _start = time.perf_counter()
    result = not crystal_graph._chain._better_than[node]

    return result


def is_bottom_node(node: str) -> bool:
    """Return True if node has no worse-than links (i.e. is a bottom node)."""
    _start = time.perf_counter()
    result = not crystal_graph._chain._worse_than[node]

    return result


def get_node_component(node: str) -> int | None:
    """Return the component id of node, or None."""
    _start = time.perf_counter()
    result = crystal_graph._chain._node_component[node]

    return result


def get_component_members(comp_id: int) -> list[str]:
    """Return the member list for a component id."""
    _start = time.perf_counter()
    result = crystal_graph._chain._component_members[comp_id]

    return result


def get_chain_length(node: str) -> int:
    """Return the chain-length metric for node."""
    _start = time.perf_counter()
    result = crystal_graph._chain._chain_length[node]

    return result


# ---------------------------------------------------------------------------
# Grouping helpers
# ---------------------------------------------------------------------------


def group_nodes_by_extreme(
    nodes: list[str],
) -> tuple[dict[int, list[str]], dict[int, list[str]]]:
    """Group nodes into top-by-component and bottom-by-component dicts.

    Returns (top_by_component, bottom_by_component) where keys are
    component ids and values are lists of node filenames.
    """
    _start = time.perf_counter()
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


def is_collapsable_pair(filename_a: str, filename_b: str) -> bool:
    """Check if a pair is collapsible (both top or both bottom in same component, no common chains)."""
    _start = time.perf_counter()
    node_a = crystal_graph.get_node(filename_a)
    node_b = crystal_graph.get_node(filename_b)
    if not node_a or not node_b:

        return False

    comp_a = crystal_graph.get_component(node_id=filename_a)
    comp_b = crystal_graph.get_component(node_id=filename_b)
    if not comp_a or not comp_b or comp_a.id != comp_b.id:

        return False

    both_top = node_a.is_top() and node_b.is_top()
    both_bottom = node_a.is_bottom() and node_b.is_bottom()

    if not (both_top or both_bottom):

        return False

    result = not crystal_graph.are_in_same_path(filename_a, filename_b)

    return result


# ---------------------------------------------------------------------------
# Filtering helpers
# ---------------------------------------------------------------------------


def filter_excluded_images(
    images: list[dict[str, Any]],
    exclude_set: set[str],
) -> list[dict[str, Any]]:
    """Remove images whose filename is in exclude_set."""
    _start = time.perf_counter()
    if not exclude_set:
        return images

    result = []
    for img in images:
        filename = img["filename"]
        if filename not in exclude_set:
            result.append(img)

    return result


def find_lowest_confidence_images(
    images: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Select a diverse subset of high-uncertainty images for fallback pairing."""
    _start = time.perf_counter()
    if not images:

        return []

    sigma = max(float(img["rating_sigma"]) for img in images)
    images = [img for img in images if float(img["rating_sigma"]) >= sigma - 0.05]
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

    candidates = (
        candidates[:section_size]
        + candidates[(n // 2 - section_size) : (n // 2 + section_size + 1)]
        + candidates[n - section_size :]
    )
    unique_candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for img in candidates:
        if img["filename"] not in seen:
            seen.add(img["filename"])
            unique_candidates.append(img)

    random.shuffle(unique_candidates)

    result = unique_candidates[:MAX_PAIR_CANDIDATES]

    return result
