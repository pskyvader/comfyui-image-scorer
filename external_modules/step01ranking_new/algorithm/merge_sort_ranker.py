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
    get_images_with_only_wins,
    get_images_with_only_losses,
)
from algorithm.confidence_tracker import calculate_confidence
from file_management.path_handler import (
    append_comparison_history_to_json,
)

# from shared.config import config
import nodes
from shared.graph import crystal_graph
from datetime import datetime, timezone

# import heapq
import math
import random
import time
import heapq
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)

_images_cache: dict[str, Any] = {"data": None, "timestamp": 0.0}
_IMAGES_CACHE_TTL = 5.5  # Cache all_images for 5.5 seconds

_MAX_PAIR_CANDIDATES = 100

# Pair type labels for debug/frontend
PAIR_TYPE_COLLAPSIBLE = "collapsible"
PAIR_TYPE_WORST_WITH_WORST = "worst_with_worst"
PAIR_TYPE_UNCONNECTED = "unconnected"
PAIR_TYPE_REFINEMENT = "refinement"
PAIR_TYPE_FALLBACK = "fallback"

# Module-level variable to store last pair selection metadata
_last_pair_metadata: dict[str, Any] = {}

from collections import OrderedDict

# --- Loop 2 Refinement Caches ---
_chain_pair_progress: dict[tuple[tuple[str, ...], tuple[str, ...]], int] = {}
_node_lru_cache: "OrderedDict[str, None]" = OrderedDict()
_chain_lru_cache: "OrderedDict[tuple[str, ...], None]" = OrderedDict()


def _update_lru(cache: OrderedDict, item: Any, max_size: int = 0):
    """Internal helper to track what was picked in this call's sub-loops."""
    if item in cache:
        cache.move_to_end(item)
    else:
        cache[item] = None
        
    if max_size > 0:
        while len(cache) > max_size:
            cache.popitem(last=False)


def get_last_pair_metadata() -> dict[str, Any]:
    """Return metadata from the most recent pair selection."""
    return _last_pair_metadata.copy()


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


# def _is_better_path(
#     candidate_edges: int,
#     candidate_impact: float,
#     current: tuple[int, float] | None,
# ) -> bool:
#     if current is None:
#         return True

#     current_edges, current_impact = current
#     return candidate_edges < current_edges or (
#         candidate_edges == current_edges and candidate_impact > current_impact
#     )


def is_collapsable_pair(filename_a: str, filename_b: str) -> bool:
    """Check if a pair is collapsible (both top or both bottom in same component, no common chains)."""
    node_a = crystal_graph.get_node(filename_a)
    node_b = crystal_graph.get_node(filename_b)
    if not node_a or not node_b:
        return False

    # Must be in the same component
    comp_a = crystal_graph.get_component(node_id=filename_a)
    comp_b = crystal_graph.get_component(node_id=filename_b)
    if not comp_a or not comp_b or comp_a.id != comp_b.id:
        return False

    # Check if both are top nodes (no better connections)
    both_top = node_a.is_top() and node_b.is_top()

    # Check if both are bottom nodes (no worse connections)
    both_bottom = node_a.is_bottom() and node_b.is_bottom()

    if not (both_top or both_bottom):
        return False

    # Must not share any chains (not in same directed path)
    return not crystal_graph.are_in_same_path(filename_a, filename_b)


# def _get_comparison_count(filename: str) -> int:
#     """Get comparison count for a node from the crystal graph."""
#     img_data = crystal_graph._images[filename]
#     if img_data:
#         return img_data["comparison_count"]
#     return 0


def select_pair_for_comparison(
    exclude_set: set[str] | None = None,
    exclude_chains: set[tuple[str, ...]] | None = None,
) -> tuple[str, str] | None:
    """
    Select the next pair of images to compare using crystal graph priorities.
      2. Worst-with-worst: components with a unique best or worst node, compare the
         remaining worst/best nodes. Sorted by least connections first.
      3. Unconnected pairs: nodes not in the same directed path.
    Final fallback: lowest confidence images from different chains.
    """
    global _last_pair_metadata
    _last_pair_metadata = {}
    total_time: float = time.time()
    step_time = time.time()

    all_images: list[dict[str, Any]] = _get_cached_all_images()
    logger.debug(
        f"[NEXT-PAIR] all images: {len(all_images)},  time: total: {time.time() - total_time}, step: {time.time() - step_time}"
    )
    step_time = time.time()

    if len(all_images) < 2:
        return None

    # Refresh crystal_graph if stale
    if crystal_graph.is_cache_stale():
        crystal_graph.rebuild_from_database(images=all_images)

    # max_height is not easily accessible in new API, using total_images for debug
    logger.debug(
        f"[NEXT-PAIR] crystal_graph built, total_images: {len(all_images)}, time: total: {time.time() - total_time}, step: {time.time() - step_time}"
    )
    step_time: float = time.time()

    # Build comparison count lookup
    comp_count_lookup: dict[str, int] = {}
    for img in all_images:
        comp_count_lookup[img["filename"]] = img["comparison_count"]

    candidate_images: list[dict[str, Any]] = _filter_excluded_images(
        all_images,
        exclude_set,
    )
    logger.debug(
        f"[NEXT-PAIR] filter_excluded_images:{len(candidate_images)} time: total: {time.time() - total_time}, step: {time.time() - step_time}"
    )
    step_time = time.time()

    if len(candidate_images) < 2:
        return None

    # Get candidate filenames for quick lookup
    candidate_filenames: set[Any] = {img["filename"] for img in candidate_images}

    pair: tuple[str, str] | None = _find_orphan_pair(
        candidate_filenames, comp_count_lookup
    )
    if pair:
        return pair

    pair = _find_extreme_pair_from_db(candidate_filenames, comp_count_lookup)
    if pair:
        return pair

    pair = _find_low_count_pairs(candidate_filenames, comp_count_lookup, all_images)
    if pair:
        return pair

    pair = _find_refinement_pairs(comp_count_lookup)
    if pair:
        return pair

    # When only one connected component remains, loop 1/2 should fully drive the
    # endgame. Falling through into broad score/fallback searches is both slow
    # and semantically wrong because it can bypass remaining structured pairs.
    if len(crystal_graph.get_all_components()) <= 1:
        return None

    pair = _find_score_range_pairs(candidate_filenames, comp_count_lookup, all_images)
    if pair:
        return pair

    pair = _find_loose_pairs(candidate_filenames, comp_count_lookup)
    if pair:
        return pair

    # Fallback
    result: tuple[str, str] | None = _find_fallback_cross_chain(
        _find_lowest_confidence_images(candidate_images)
    )
    if result:
        a, b = result
        logger.info(
            f"[NEXT-PAIR] Loop=fallback Pair={result} LeftComparisonCount={comp_count_lookup[a]} RightComparisonCount={comp_count_lookup[b]}"
        )
    return result


def _find_orphan_pair(
    candidate_filenames: set[str], comp_count_lookup: dict[str, int]
) -> tuple[str, str] | None:
    """Loop 0: Find orphans (images with 0 comparisons)."""
    global _last_pair_metadata
    orphan_nodes: list[str] = [
        n for n in candidate_filenames if comp_count_lookup[n] == 0
    ]
    logger.debug(f"orphan nodes: {len(orphan_nodes)}")
    if len(orphan_nodes) >= 2:
        pair = (orphan_nodes[0], orphan_nodes[1])
        _last_pair_metadata = {
            "pair_type": "orphan",
            "chain_level": -1,
            "component_size": 0,
            "left_comp_count": 0,
            "right_comp_count": 0,
        }
        logger.info(
            f"[NEXT-PAIR] Loop=0 Type=orphan Pair={pair} LeftComparisonCount=0 RightComparisonCount=0"
        )
        return pair
    return None


def _find_extreme_pair_from_db(
    candidate_filenames: set[str],
    comp_count_lookup: dict[str, int],
) -> tuple[str, str] | None:
    """Loop 0.5: Pair pure winners or pure losers queried directly from the database.

    Looks at the actual comparison records to find images that have ONLY ever won
    (never lost) or ONLY ever lost (never won), regardless of what the in-memory
    graph currently reflects. These are the most extreme candidates and need to be
    compared against each other to collapse the ranking.

    Priority: pair same-extreme-type images with the least comparison counts first.
    """
    global _last_pair_metadata

    only_wins = [n for n in get_images_with_only_wins() if n in candidate_filenames]
    only_losses = [n for n in get_images_with_only_losses() if n in candidate_filenames]

    if len(only_wins) >= 2:
        only_wins.sort(key=lambda n: comp_count_lookup[n])
        pair = (only_wins[0], only_wins[1])
        _last_pair_metadata = {
            "pair_type": "extreme_wins",
            "chain_level": -1,
            "component_size": 0,
            "left_comp_count": comp_count_lookup[pair[0]],
            "right_comp_count": comp_count_lookup[pair[1]],
        }
        logger.info(
            f"[NEXT-PAIR] Loop=0.5 Type=pure_winners Pair={pair} "
            f"LeftComparisonCount={comp_count_lookup[pair[0]]} "
            f"RightComparisonCount={comp_count_lookup[pair[1]]}"
        )
        return pair

    if len(only_losses) >= 2:
        only_losses.sort(key=lambda n: comp_count_lookup[n])
        pair = (only_losses[0], only_losses[1])
        _last_pair_metadata = {
            "pair_type": "extreme_losses",
            "chain_level": -1,
            "component_size": 0,
            "left_comp_count": comp_count_lookup[pair[0]],
            "right_comp_count": comp_count_lookup[pair[1]],
        }
        logger.info(
            f"[NEXT-PAIR] Loop=0.5 Type=pure_losers Pair={pair} "
            f"LeftComparisonCount={comp_count_lookup[pair[0]]} "
            f"RightComparisonCount={comp_count_lookup[pair[1]]}"
        )
        return pair

    if only_wins and only_losses:
        pair = (only_wins[0], only_losses[0])
        _last_pair_metadata = {
            "pair_type": "extreme_mixed",
            "chain_level": -1,
            "component_size": 0,
            "left_comp_count": comp_count_lookup[pair[0]],
            "right_comp_count": comp_count_lookup[pair[1]],
        }
        logger.info(
            f"[NEXT-PAIR] Loop=0.5 Type=pure_winner_vs_pure_loser Pair={pair} "
            f"LeftComparisonCount={comp_count_lookup[pair[0]]} "
            f"RightComparisonCount={comp_count_lookup[pair[1]]}"
        )
        return pair

    return None


def _find_low_count_pairs(
    candidate_filenames: set[str],
    comp_count_lookup: dict[str, int],
    all_images: list[dict],
) -> tuple[str, str] | None:
    """Loop 1: Low comparison count - Comparison count -> Component size -> Chain height -> Collapsible/Merge."""
    global _last_pair_metadata
    comparison_counts: list[int] = sorted(
        c for c in set(comp_count_lookup.values()) if c > 0
    )
    if len(crystal_graph._chain._component_members) <= 1:
        chain_heights: list[int] = sorted(crystal_graph._chain._nodes_by_length.keys())
        extreme_nodes = [
            n
            for n in candidate_filenames
            if not crystal_graph.get_node(n).get_links(better_than=True)
            or not crystal_graph.get_node(n).get_links(worse_than=True)
        ]
        component_size = (
            len(next(iter(crystal_graph._chain._component_members.values())))
            if crystal_graph._chain._component_members
            else len(candidate_filenames)
        )
        for target_comparison_count in comparison_counts:
            nodes_at_or_below_target = [
                n
                for n in extreme_nodes
                if comp_count_lookup[n] <= target_comparison_count
            ]
            if not nodes_at_or_below_target:
                continue

            for target_chain_height in chain_heights:
                nodes_at_chain = [
                    n
                    for n in nodes_at_or_below_target
                    if crystal_graph._chain._chain_length[n] <= target_chain_height
                ]
                if not nodes_at_chain:
                    continue

                for prefer_bottom in (True, False):
                    pair = _find_collapsable_pair_by_extremes(
                        nodes_at_chain,
                        -1,
                        comp_count_lookup,
                        prefer_bottom=prefer_bottom,
                        target_count=target_comparison_count,
                        candidate_filenames=candidate_filenames,
                    )
                    if not pair:
                        continue

                    _last_pair_metadata = {
                        "pair_type": PAIR_TYPE_COLLAPSIBLE,
                        "chain_level": target_chain_height,
                        "component_size": component_size,
                        "left_comp_count": comp_count_lookup[pair[0]],
                        "right_comp_count": comp_count_lookup[pair[1]],
                    }
                    logger.info(
                        f"[NEXT-PAIR] Loop=1 Sub=single_component_collapsible "
                        f"ComparisonCount={target_comparison_count} "
                        f"ChainHeight={target_chain_height} Pair={pair} "
                        f"LeftComparisonCount={comp_count_lookup[pair[0]]} "
                        f"RightComparisonCount={comp_count_lookup[pair[1]]}"
                    )
                    return pair
        return None

    component_sizes: dict[int, int] = {
        comp.id: comp.size for comp in crystal_graph.get_all_components()
    }
    sorted_component_sizes: list[int] = sorted(set(component_sizes.values()))

    chain_heights: list[int] = sorted(crystal_graph._chain._nodes_by_length.keys())
    logger.debug(f"""
        [NEXT-PAIR] Loop=1 
        Comparison counts: {comparison_counts}
        Component sizes: {len(component_sizes)}
        Sorted component sizes: {len(sorted_component_sizes)}
        Chain heights: {len(chain_heights)}"
    """)
    for target_comparison_count in comparison_counts:
        # Step 1: Get all nodes with count <= target_count
        nodes_up_to_count = [
            n
            for n in candidate_filenames
            if comp_count_lookup[n] <= target_comparison_count
        ]

        # Step 2: Filter to only extremes (top/bottom nodes only)
        top_nodes: list[str] = [
            n for n in nodes_up_to_count if not crystal_graph._chain._better_than[n]
        ]
        bottom_nodes: list[str] = [
            n for n in nodes_up_to_count if not crystal_graph._chain._worse_than[n]
        ]
        if len(top_nodes) > 0 or len(bottom_nodes) > 0:
            logger.debug(
                f"[NEXT-PAIR] Loop=1 Nodes up to count {target_comparison_count}: {len(nodes_up_to_count)}, top nodes: {len(top_nodes)}, bottom nodes: {len(bottom_nodes)}"
            )
        extreme_nodes = top_nodes + bottom_nodes
        # if len(extreme_nodes) < 2:
        #     continue

        # Step 3: Loop count -> component -> chain -> pair type
        for target_component_size in sorted_component_sizes:
            nodes_at_component = [
                n
                for n in extreme_nodes
                if crystal_graph.get_component(node_id=n).id <= target_component_size
            ]
            if len(nodes_at_component) > 0:
                logger.debug(
                    f"nodes at component {target_component_size}: {len(nodes_at_component)}"
                )
            # if len(nodes_at_component) < 2:
            #     continue

            # Separate nodes at or below target count from nodes with higher count
            nodes_at_or_below_target = [
                n
                for n in nodes_at_component
                if comp_count_lookup[n] <= target_comparison_count
            ]

            # PHASE A: Same-level collapsible pairs (nodes at or below target count, same extreme)
            if len(nodes_at_or_below_target) > 0:
                logger.debug(
                    f"PHASE A: nodes_at_or_below_target count: {len(nodes_at_or_below_target)}"
                )
            # logger.debug(
            #     f"PHASE A: nodes_at_or_below_target: {nodes_at_or_below_target}"
            # )
            # logger.debug(
            #     f"PHASE A: top nodes in component: {len([n for n in nodes_at_or_below_target if not crystal_graph._chain._better_than[n]])}"
            # )
            # logger.debug(
            #     f"PHASE A: bottom nodes in component: {len([n for n in nodes_at_or_below_target if not crystal_graph._chain._worse_than[n]])}"
            # )
            if len(nodes_at_or_below_target) >= 1:
                for target_chain_height in chain_heights:
                    nodes_at_chain = [
                        n
                        for n in nodes_at_or_below_target
                        if crystal_graph._chain._chain_length[n] <= target_chain_height
                    ]
                    # logger.debug(
                    #     f"PHASE A: height={target_chain_height}, nodes_at_chain count: {len(nodes_at_chain)}"
                    # )
                    # if len(nodes_at_chain) < 2:
                    #     continue

                    # Try collapsible bottom first
                    pair = _find_collapsable_pair_by_extremes(
                        nodes_at_chain,
                        -1,
                        comp_count_lookup,
                        prefer_bottom=True,
                        target_count=target_comparison_count,
                        candidate_filenames=candidate_filenames,
                    )
                    # logger.debug(f"collapsible pair: {pair}")
                    if pair:
                        comp_a = crystal_graph._chain._node_component[pair[0]]
                        comp_b = crystal_graph._chain._node_component[pair[1]]
                        same_comp = "same" if comp_a == comp_b else "diff"
                        _last_pair_metadata = {
                            "pair_type": PAIR_TYPE_COLLAPSIBLE,
                            "chain_level": target_chain_height,
                            "component_size": target_component_size,
                            "left_comp_count": comp_count_lookup[pair[0]],
                            "right_comp_count": comp_count_lookup[pair[1]],
                        }
                        logger.info(
                            f"[NEXT-PAIR] Loop=1 Sub=collapsible_bottom ComparisonCount={target_comparison_count} ComponentSize={target_component_size} ChainHeight={target_chain_height} Pair={pair} SameComponent={same_comp} LeftComparisonCount={comp_count_lookup[pair[0]]} RightComparisonCount={comp_count_lookup[pair[1]]}"
                        )
                        return pair

                    # Try collapsible top
                    pair = _find_collapsable_pair_by_extremes(
                        nodes_at_chain,
                        -1,
                        comp_count_lookup,
                        prefer_bottom=False,
                        target_count=target_comparison_count,
                        candidate_filenames=candidate_filenames,
                    )
                    if pair:
                        comp_a = crystal_graph._chain._node_component[pair[0]]
                        comp_b = crystal_graph._chain._node_component[pair[1]]
                        same_comp = "same" if comp_a == comp_b else "diff"
                        _last_pair_metadata = {
                            "pair_type": PAIR_TYPE_COLLAPSIBLE,
                            "chain_level": target_chain_height,
                            "component_size": target_component_size,
                            "left_comp_count": comp_count_lookup[pair[0]],
                            "right_comp_count": comp_count_lookup[pair[1]],
                        }
                        logger.info(
                            f"[NEXT-PAIR] Loop=1 Sub=collapsible_top ComparisonCount={target_comparison_count} ComponentSize={target_component_size} ChainHeight={target_chain_height} Pair={pair} SameComponent={same_comp} LeftComparisonCount={comp_count_lookup[pair[0]]} RightComparisonCount={comp_count_lookup[pair[1]]}"
                        )
                        return pair

            # PHASE B: Cross-level pairs - low count extreme paired with higher count node in same component
            higher_nodes = []
            components_to_check = set(
                crystal_graph._chain._node_component[n] for n in nodes_at_component
            )
            for comp_id in components_to_check:
                comp_members = crystal_graph._chain._component_members[comp_id]
                for member in comp_members:
                    if (
                        member in candidate_filenames
                        and comp_count_lookup[member] > target_comparison_count
                    ):
                        higher_nodes.append(member)
            if nodes_at_or_below_target and higher_nodes:
                logger.debug(
                    f"PHASE B: Cross-level nodes at/below:{len(nodes_at_or_below_target)}, higher nodes: {len(higher_nodes)}"
                )
                for node in nodes_at_or_below_target:
                    node_comp = crystal_graph._chain._node_component[node]
                    node_is_bottom = not crystal_graph._chain._worse_than[node]
                    node_is_top = not crystal_graph._chain._better_than[node]
                    if not node_is_bottom and not node_is_top:
                        continue
                    node_chain = crystal_graph._chain._chain_length[node]
                    all_component_nodes = crystal_graph._chain._component_members[
                        node_comp
                    ]
                    partner_candidates = [
                        n
                        for n in all_component_nodes
                        if n != node
                        and n in candidate_filenames
                        and comp_count_lookup[n] > target_comparison_count
                        and not crystal_graph.are_in_same_path(node, n)
                    ]
                    for partner in partner_candidates:
                        partner_is_bottom = not crystal_graph._chain._worse_than[
                            partner
                        ]
                        partner_is_top = not crystal_graph._chain._better_than[partner]
                        same_extreme = (node_is_bottom and partner_is_bottom) or (
                            node_is_top and partner_is_top
                        )
                        if same_extreme:
                            pair = (node, partner)
                            _last_pair_metadata = {
                                "pair_type": "cross_level_pair",
                                "chain_level": node_chain,
                                "component_size": target_component_size,
                                "left_comp_count": comp_count_lookup[pair[0]],
                                "right_comp_count": comp_count_lookup[pair[1]],
                            }
                            logger.info(
                                f"[NEXT-PAIR] Loop=1 Sub=cross_level_pair ComparisonCount={target_comparison_count} ComponentSize={target_component_size} ChainHeight={node_chain} Pair={pair} LeftComparisonCount={comp_count_lookup[pair[0]]} RightComparisonCount={comp_count_lookup[pair[1]]}"
                            )
                            return pair

            # PHASE C: Same-level merge pairs (different components)
            for target_chain_height in chain_heights:
                nodes_at_chain = [
                    n
                    for n in nodes_at_or_below_target
                    if crystal_graph._chain._chain_length[n] <= target_chain_height
                ]
                if not nodes_at_chain:
                    continue

                if len(nodes_at_chain) >= 2:
                    pair = _find_merging_pair_by_extremes(
                        nodes_at_chain,
                        target_chain_height,
                        comp_count_lookup,
                        prefer_bottom=True,
                    )
                    if pair:
                        comp_a = crystal_graph._chain._node_component[pair[0]]
                        comp_b = crystal_graph._chain._node_component[pair[1]]
                        _last_pair_metadata = {
                            "pair_type": PAIR_TYPE_WORST_WITH_WORST,
                            "chain_level": target_chain_height,
                            "component_size": target_component_size,
                            "left_comp_count": comp_count_lookup[pair[0]],
                            "right_comp_count": comp_count_lookup[pair[1]],
                        }
                        logger.info(
                            f"[NEXT-PAIR] Loop=1 Sub=merge_bottom ComparisonCount={target_comparison_count} ComponentSize={target_component_size} ChainHeight={target_chain_height} Pair={pair} ComponentA={comp_a} ComponentB={comp_b} LeftComparisonCount={comp_count_lookup[pair[0]]} RightComparisonCount={comp_count_lookup[pair[1]]}"
                        )
                        return pair

                    pair = _find_merging_pair_by_extremes(
                        nodes_at_chain,
                        target_chain_height,
                        comp_count_lookup,
                        prefer_bottom=False,
                    )
                    if pair:
                        comp_a = crystal_graph._chain._node_component[pair[0]]
                        comp_b = crystal_graph._chain._node_component[pair[1]]
                        _last_pair_metadata = {
                            "pair_type": PAIR_TYPE_WORST_WITH_WORST,
                            "chain_level": target_chain_height,
                            "component_size": target_component_size,
                            "left_comp_count": comp_count_lookup[pair[0]],
                            "right_comp_count": comp_count_lookup[pair[1]],
                        }
                        logger.info(
                            f"[NEXT-PAIR] Loop=1 Sub=merge_top ComparisonCount={target_comparison_count} ComponentSize={target_component_size} ChainHeight={target_chain_height} Pair={pair} ComponentA={comp_a} ComponentB={comp_b} LeftComparisonCount={comp_count_lookup[pair[0]]} RightComparisonCount={comp_count_lookup[pair[1]]}"
                        )
                        return pair

                exact_target_nodes = [
                    n
                    for n in nodes_at_chain
                    if comp_count_lookup[n] == target_comparison_count
                ]
                pair = _find_cross_component_progression_pair(
                    exact_target_nodes,
                    candidate_filenames,
                    comp_count_lookup,
                    target_comparison_count,
                )
                if pair:
                    comp_a = crystal_graph._chain._node_component[pair[0]]
                    comp_b = crystal_graph._chain._node_component[pair[1]]
                    _last_pair_metadata = {
                        "pair_type": "cross_component_progression",
                        "chain_level": target_chain_height,
                        "component_size": target_component_size,
                        "left_comp_count": comp_count_lookup[pair[0]],
                        "right_comp_count": comp_count_lookup[pair[1]],
                    }
                    logger.info(
                        f"[NEXT-PAIR] Loop=1 Sub=cross_component_progression ComparisonCount={target_comparison_count} ComponentSize={target_component_size} ChainHeight={target_chain_height} Pair={pair} ComponentA={comp_a} ComponentB={comp_b} LeftComparisonCount={comp_count_lookup[pair[0]]} RightComparisonCount={comp_count_lookup[pair[1]]}"
                    )
                    return pair
    return None


def _get_pair_from_chains(
    chain_a: list[str], chain_b: list[str], comp_count_lookup: dict[str, int]
) -> tuple[str, str] | None:
    """Select a pair of nodes from two chains by identifying the largest gaps between shared points."""
    key = (tuple(chain_a), tuple(chain_b))
    next_idx = _chain_pair_progress.get(key, 0)

    # 1. Find shared nodes and their indices
    cb_set = set(chain_b)
    common = [n for n in chain_a if n in cb_set]

    idx_a = {n: i for i, n in enumerate(chain_a)}
    idx_b = {n: i for i, n in enumerate(chain_b)}

    # 2. Identify waypoints (shared nodes) and boundary points
    waypoints = []
    for n in common:
        waypoints.append((idx_a[n], idx_b[n]))

    # If they don't share start/end, add them as boundaries
    if not waypoints:
        waypoints = [(0, 0), (len(chain_a) - 1, len(chain_b) - 1)]
    else:
        # Ensure we cover the head and tail of the chains
        if waypoints[0][0] > 0 or waypoints[0][1] > 0:
            waypoints.insert(0, (0, 0))
        if waypoints[-1][0] < len(chain_a) - 1 or waypoints[-1][1] < len(chain_b) - 1:
            waypoints.append((len(chain_a) - 1, len(chain_b) - 1))

    # 3. Identify gaps between waypoints
    gaps = []
    for i in range(len(waypoints) - 1):
        w1 = waypoints[i]
        w2 = waypoints[i + 1]

        # Gap size is the number of unshared nodes in both chains
        size_a = w2[0] - w1[0] - 1
        size_b = w2[1] - w1[1] - 1

        if size_a > 0 or size_b > 0:
            gaps.append({"w1": w1, "w2": w2, "size": size_a + size_b})

    if not gaps:
        return None

    # 4. Pick node in the largest gap
    gaps.sort(key=lambda x: x["size"], reverse=True)

    # Use next_idx to rotate through gaps if needed (simple rotation for now)
    gap = gaps[next_idx % len(gaps)]
    _chain_pair_progress[key] = next_idx + 1

    w1, w2 = gap["w1"], gap["w2"]

    # Pick a node from the gap (trying to find an uncompared one or just the middle)
    if (w2[0] - w1[0]) >= (w2[1] - w1[1]):
        target_idx = w1[0] + (w2[0] - w1[0]) // 2
        node_a = chain_a[target_idx]
        target_idx_b = w1[1] + (w2[1] - w1[1]) // 2
        node_b = chain_b[target_idx_b]
        return (node_a, node_b)
    else:
        target_idx = w1[1] + (w2[1] - w1[1]) // 2
        node_b = chain_b[target_idx]
        target_idx_a = w1[0] + (w2[0] - w1[0]) // 2
        node_a = chain_a[target_idx_a]
        return (node_a, node_b)


def _get_prioritized_chain_pairs() -> tuple[list[str], list[str]] | None:
    """Find the two shortest chains in the graph that are not in the LRU cache.
    Returns a tuple of (chain_a, chain_b) or None.
    """
    start_time = time.time()

    # 1. Get all chains from the graph
    all_chain_proxies = crystal_graph.get_all_chains()
    if not all_chain_proxies or len(all_chain_proxies) < 2:
        return None

    # Maintain LRU cache size: max 1/4 of total chains so 3/4 remain available
    max_chain_cache = max(2, len(all_chain_proxies) // 4)
    while len(_chain_lru_cache) > max_chain_cache:
        _chain_lru_cache.popitem(last=False)

    # 2. Extract raw nodes and filter by LRU
    # PERFORMANCE: Accessing _nodes directly to avoid proxy overhead
    available_chains = []
    for proxy in all_chain_proxies:
        nodes = proxy._nodes
        if not nodes:
            continue

        # Check if this chain (as a tuple) is in our recently used cache
        if tuple(nodes) in _chain_lru_cache:
            continue

        available_chains.append(nodes)

    if len(available_chains) < 2:
        # If we filtered too many, just take the shortest ones regardless of LRU
        # (to avoid getting stuck), or just return None if we really want strict LRU.
        # Following "must be the 2 shortest chains (that are not on the lru chain cache)"
        return None

    # 3. Sort by length (shortest first)
    available_chains.sort(key=len)

    # 4. Return the 2 shortest
    res = (available_chains[0], available_chains[1])

    logger.debug(
        f"[TIMER] _get_prioritized_chain_pairs (Simplified) took {time.time() - start_time:.4f}s"
    )
    return res


def _find_valid_refinement_pair(
    chain_pair: tuple[list[str], list[str]], comp_count_lookup: dict[str, int]
) -> tuple[str, str] | None:
    """Part 2 of Loop 2: Get a node pair from the selected two chains."""
    global _last_pair_metadata
    start_time = time.time()

    ca, cb = chain_pair
    pair = _get_pair_from_chains(ca, cb, comp_count_lookup)
    if pair:
        # Successfully found a refinement pair
        _update_lru(_chain_lru_cache, tuple(ca))
        _update_lru(_chain_lru_cache, tuple(cb))
        _update_lru(_node_lru_cache, pair[0])
        _update_lru(_node_lru_cache, pair[1])

        _last_pair_metadata = {
            "pair_type": PAIR_TYPE_REFINEMENT,
            "chain_a": tuple(ca),
            "chain_b": tuple(cb),
            "chain_len_a": len(ca),
            "chain_len_b": len(cb),
            "left_comp_count": comp_count_lookup.get(pair[0], 0),
            "right_comp_count": comp_count_lookup.get(pair[1], 0),
        }
        logger.info(
            f"[NEXT-PAIR] Loop=2 Sub=chain_refinement "
            f"ChainLengths=({len(ca)},{len(cb)}) "
            f"Pair={pair} "
            f"LeftCompCount={comp_count_lookup.get(pair[0], 0)} "
            f"RightCompCount={comp_count_lookup.get(pair[1], 0)}"
        )
        logger.debug(
            f"[TIMER] _find_valid_refinement_pair took {time.time() - start_time:.4f}s"
        )
        return pair

    logger.debug(
        f"[TIMER] _find_valid_refinement_pair (no gap) took {time.time() - start_time:.4f}s"
    )
    return None


def _find_refinement_pairs(comp_count_lookup: dict[str, int]) -> tuple[str, str] | None:
    """Loop 2: Chain refinement within a single graph component."""
    start_time = time.time()

    chain_pair = _get_prioritized_chain_pairs()
    if not chain_pair:
        return None

    pair = _find_valid_refinement_pair(chain_pair, comp_count_lookup)

    logger.debug(
        f"[TIMER] _find_refinement_pairs TOTAL took {time.time() - start_time:.4f}s"
    )
    return pair


def _find_score_range_pairs(
    candidate_filenames: set[str],
    comp_count_lookup: dict[str, int],
    all_images: list[dict],
) -> tuple[str, str] | None:
    """Loop 3: Score range - Expand from middle (0.5) outward."""
    global _last_pair_metadata
    nodes_by_score = {
        n: img["score"]
        for n in candidate_filenames
        if (img := next((i for i in all_images if i["filename"] == n), None))
    }
    middle_score = 0.5
    for score_range_expand in [
        0.0,
        0.05,
        0.1,
        0.15,
        0.2,
        0.25,
        0.3,
        0.35,
        0.4,
        0.45,
        0.5,
    ]:
        min_score = middle_score - score_range_expand
        max_score = middle_score + score_range_expand
        nodes_within_score_range = [
            n
            for n in candidate_filenames
            if min_score <= nodes_by_score[n] <= max_score
        ]
        if len(nodes_within_score_range) < 2:
            continue
        nodes_within_score_range.sort(
            key=lambda n: (
                comp_count_lookup[n],
                abs(nodes_by_score[n] - middle_score),
            )
        )
        for i in range(len(nodes_within_score_range)):
            for j in range(i + 1, len(nodes_within_score_range)):
                a, b = nodes_within_score_range[i], nodes_within_score_range[j]
                if (
                    not crystal_graph.are_in_same_path(a, b)
                    or crystal_graph._chain._node_component[a] is None
                ):
                    pair = (a, b)
                    score_a = nodes_by_score[a]
                    score_b = nodes_by_score[b]
                    comp_a = crystal_graph._chain._node_component[a]
                    comp_b = crystal_graph._chain._node_component[b]
                    same_comp = (
                        "same"
                        if comp_a == comp_b
                        else "diff"
                        if comp_a and comp_b
                        else "none"
                    )
                    _last_pair_metadata = {
                        "pair_type": "score_range",
                        "chain_level": -1,
                        "component_size": -1,
                        "left_comp_count": comp_count_lookup[a],
                        "right_comp_count": comp_count_lookup[b],
                    }
                    logger.info(
                        f"[NEXT-PAIR] Loop=3 Sub=score_range ScoreRange={min_score:.2f}-{max_score:.2f} Pair={pair} SameComponent={same_comp} LeftScore={score_a:.3f} RightScore={score_b:.3f} LeftComparisonCount={comp_count_lookup[a]} RightComparisonCount={comp_count_lookup[b]}"
                    )
                    return pair
    return None


def _find_loose_pairs(
    candidate_filenames: set[str], comp_count_lookup: dict[str, int]
) -> tuple[str, str] | None:
    """Loop 4: Loose pairs - Any loose nodes with same or lower comparison count."""
    global _last_pair_metadata
    if not comp_count_lookup:
        return None
    max_comparison_count = max(comp_count_lookup.values())
    for target_comparison_count in range(1, max_comparison_count + 1):
        nodes_with_comparison_count = [
            n
            for n in candidate_filenames
            if comp_count_lookup[n] <= target_comparison_count
        ]
        if len(nodes_with_comparison_count) >= 2:
            nodes_with_comparison_count.sort(key=lambda n: comp_count_lookup[n])
            pair = (nodes_with_comparison_count[0], nodes_with_comparison_count[1])
            comp_a = crystal_graph._chain._node_component[pair[0]]
            comp_b = crystal_graph._chain._node_component[pair[1]]
            same_comp = (
                "same" if comp_a == comp_b else "diff" if comp_a and comp_b else "none"
            )
            _last_pair_metadata = {
                "pair_type": "loose_final",
                "chain_level": -1,
                "component_size": -1,
                "left_comp_count": comp_count_lookup[pair[0]],
                "right_comp_count": comp_count_lookup[pair[1]],
            }
            logger.info(
                f"[NEXT-PAIR] Loop=4 Sub=loose_final MaxComparisonCount={target_comparison_count} Pair={pair} SameComponent={same_comp} LeftComparisonCount={comp_count_lookup[pair[0]]} RightComparisonCount={comp_count_lookup[pair[1]]}"
            )
            return pair
    return None


def _filter_excluded_images(
    images: list[dict[str, Any]],
    exclude_set: set[str] | None,
) -> list[dict[str, Any]]:
    if not exclude_set or len(images) <= 5:
        return images

    safe_exclude = set(exclude_set)
    result = []
    for img in images:
        filename = img["filename"]
        if filename not in safe_exclude:
            result.append(img)
    return result


def _get_comparison_graph(all_images: list[dict[str, Any]]) -> "CrystalGraph":
    """Step 3: Get comparison graph data needed for filtering. Uses the global crystal_graph instance."""
    if crystal_graph.is_cache_stale():
        crystal_graph.rebuild_from_database(images=all_images, include_transitive=False)
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
    nodes_at_height: list[str], height: int, comp_count_lookup: dict[str, int]
) -> tuple[str, str] | None:
    """Find collapsable pairs at a given height (multiple top/bottom nodes in same component).

    Priority: pairs with the least total comparison count first.
    Both nodes should ideally have similar comparison counts.
    """
    # Group top and bottom nodes by component
    top_by_component: dict[int, list[str]] = defaultdict(list)
    bottom_by_component: dict[int, list[str]] = defaultdict(list)

    for n in nodes_at_height:
        comp_id = crystal_graph._chain._node_component[n]
        if comp_id is None:
            continue
        if not crystal_graph._chain._better_than[n]:
            top_by_component[comp_id].append(n)
        if not crystal_graph._chain._worse_than[n]:
            bottom_by_component[comp_id].append(n)

    # Collect all candidate collapsible pairs with their priority score
    candidate_pairs: list[tuple[int, int, int, str, str]] = []

    for groups in [bottom_by_component, top_by_component]:
        for comp_id, group in groups.items():
            if len(group) < 2:
                continue
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    a, b = group[i], group[j]
                    if not crystal_graph.are_in_same_path(a, b):
                        count_a = comp_count_lookup[a]
                        count_b = comp_count_lookup[b]
                        # Sort key: (min_count, total_count, count_difference) — prioritize smaller counts first
                        min_count = min(count_a, count_b)
                        total_count = count_a + count_b
                        count_diff = abs(count_a - count_b)
                        candidate_pairs.append(
                            (min_count, total_count, count_diff, a, b)
                        )

    if not candidate_pairs:
        logger.debug(f"[COLLAPSABLE-PAIR] No collapsible pairs at height {height}")
        return None

    # Sort by min_count first (smaller counts get priority), then total_count, then count_diff
    candidate_pairs.sort(key=lambda x: (x[0], x[1], x[2]))

    best = candidate_pairs[0]
    logger.debug(
        f"[COLLAPSABLE-PAIR] Found {len(candidate_pairs)} collapsible pairs at height {height}. "
        f"Best: {best[3]} vs {best[4]} (min_count={best[0]}, total={best[1]}, diff={best[2]})"
    )
    return (best[3], best[4])


# def _find_merging_pair(
#     nodes_at_height: list[str], height: int, comp_count_lookup: dict[str, int]
# ) -> tuple[str, str] | None:
#     """Find pairs of best/worst nodes from DIFFERENT components to merge them.

#     IMPORTANT: Only match same extreme type - bottoms with bottoms, tops with tops.
#     Never mix bottom with top.

#     Goal: Connect unconnected or sparsely connected hierarchies.
#     Priority: least connections first.
#     """
#     node_set = set(nodes_at_height)

#     # Group available nodes by component
#     comp_to_nodes: dict[int, list[str]] = defaultdict(list)
#     for n in nodes_at_height:
#         comp_id = crystal_graph._chain._node_component[n]
#         if comp_id is not None:
#             comp_to_nodes[comp_id].append(n)

#     comp_ids = list(comp_to_nodes.keys())
#     if len(comp_ids) < 2:
#         return None

#     candidate_pairs: list[tuple[int, int, str, str]] = []

#     def get_extremes(cid):
#         members = crystal_graph._chain._component_members.get(cid, [])
#         tops = [
#             n for n in members if n in node_set and not crystal_graph._chain._better_than[n]
#         ]
#         bottoms = [
#             n for n in members if n in node_set and not crystal_graph._chain._worse_than[n]
#         ]
#         return tops, bottoms

#     for i in range(len(comp_ids)):
#         for j in range(i + 1, len(comp_ids)):
#             comp_a = comp_ids[i]
#             comp_b = comp_ids[j]

#             tops_a, bottoms_a = get_extremes(comp_a)
#             tops_b, bottoms_b = get_extremes(comp_b)

#             for a in tops_a:
#                 for b in tops_b:
#                     count_a = comp_count_lookup[a]
#                     count_b = comp_count_lookup[b]
#                     candidate_pairs.append(
#                         (count_a + count_b, abs(count_a - count_b), a, b)
#                     )

#             for a in bottoms_a:
#                 for b in bottoms_b:
#                     count_a = comp_count_lookup[a]
#                     count_b = comp_count_lookup[b]
#                     candidate_pairs.append(
#                         (count_a + count_b, abs(count_a - count_b), a, b)
#                     )

#     if not candidate_pairs:
#         return None

#     candidate_pairs.sort()
#     best = candidate_pairs[0]
#     return (best[2], best[3])


def _find_collapsable_pair_by_extremes(
    nodes_at_height: list[str],
    height: int,
    comp_count_lookup: dict[str, int],
    prefer_bottom: bool,
    target_count: int | None = None,
    candidate_filenames: set[str] | None = None,
) -> tuple[str, str] | None:
    """Find collapsible pairs with preference for bottom or top nodes.

    If target_count is provided, check each component:
    - If there are 2+ nodes at target_count level -> pair them (same level)
    - If there's only 1 node at target_count -> pair with higher count node in same component (cross-level)
    """
    top_by_component: dict[int, list[str]] = defaultdict(list)
    bottom_by_component: dict[int, list[str]] = defaultdict(list)

    for n in nodes_at_height:
        comp_id = crystal_graph._chain._node_component[n]
        if comp_id is None:
            continue
        if not crystal_graph._chain._better_than[n]:
            top_by_component[comp_id].append(n)
        if not crystal_graph._chain._worse_than[n]:
            bottom_by_component[comp_id].append(n)

    # Use requested extreme first
    groups = (
        [bottom_by_component, top_by_component]
        if prefer_bottom
        else [top_by_component, bottom_by_component]
    )

    candidate_pairs: list[tuple[int, int, int, int, str, str]] = []

    for group_dict in groups:
        for comp_id, group in group_dict.items():
            # logger.debug(
            #     f"collapsible: comp_id={comp_id}, group_size={len(group)}, group={group}"
            # )
            # if len(group) < 2:
            #     continue

            # Separate nodes by comparison count
            nodes_by_count: dict[int, list[str]] = defaultdict(list)
            for n in group:
                count = comp_count_lookup[n]
                nodes_by_count[count].append(n)

            if target_count is not None:
                target_nodes = nodes_by_count[target_count]

                # Case 1: 2+ nodes at target level -> pair same level
                if len(target_nodes) >= 2:
                    for i in range(len(target_nodes)):
                        for j in range(i + 1, len(target_nodes)):
                            a, b = target_nodes[i], target_nodes[j]
                            if not crystal_graph.are_in_same_path(a, b):
                                count_a = comp_count_lookup[a]
                                count_b = comp_count_lookup[b]
                                candidate_pairs.append(
                                    (
                                        1,
                                        abs(target_count - count_a),
                                        abs(target_count - count_b),
                                        abs(count_a - count_b),
                                        a,
                                        b,
                                    )
                                )

                # Case 2: Only 1 node at target level -> pair with higher level in same component
                elif len(target_nodes) == 1:
                    target_node = target_nodes[0]
                    if group_dict is bottom_by_component:
                        same_extreme_higher_nodes = [
                            n
                            for n in crystal_graph._chain._component_members[comp_id]
                            if n != target_node
                            and (
                                candidate_filenames is None or n in candidate_filenames
                            )
                            and comp_count_lookup[n] > target_count
                            and not crystal_graph._chain._worse_than[n]
                        ]
                    else:
                        same_extreme_higher_nodes = [
                            n
                            for n in crystal_graph._chain._component_members[comp_id]
                            if n != target_node
                            and (
                                candidate_filenames is None or n in candidate_filenames
                            )
                            and comp_count_lookup[n] > target_count
                            and not crystal_graph._chain._better_than[n]
                        ]

                    same_extreme_higher_nodes.sort(
                        key=lambda n: (
                            comp_count_lookup[n],
                            abs(comp_count_lookup[n] - comp_count_lookup[target_node]),
                        )
                    )
                    for higher_node in same_extreme_higher_nodes:
                        if not crystal_graph.are_in_same_path(target_node, higher_node):
                            count_a: int = comp_count_lookup[target_node]
                            count_b: int = comp_count_lookup[higher_node]
                            candidate_pairs.append(
                                (
                                    0,
                                    count_a,
                                    count_b,
                                    abs(count_a - count_b),
                                    target_node,
                                    higher_node,
                                )
                            )

                # When target_count is set, the outer selector loop already
                # iterates comparison counts in ascending order. Falling back to
                # other counts here is redundant and can cause endgame loops by
                # reintroducing higher-level collapsible work inside the current
                # target-count pass.
                continue

            # Fallback: same-level pairs for other counts
            for count, nodes in nodes_by_count.items():
                logger.debug(f"collapsible fallback: count={count}, nodes={len(nodes)}")
                for i in range(len(nodes)):
                    for j in range(i + 1, len(nodes)):
                        a, b = nodes[i], nodes[j]
                        if not crystal_graph.are_in_same_path(a, b):
                            count_a = comp_count_lookup[a]
                            count_b = comp_count_lookup[b]
                            candidate_pairs.append(
                                (2, count_a, count_b, abs(count_a - count_b), a, b)
                            )

    if not candidate_pairs:
        # logger.debug(
        #     f"[COLLAPSABLE] No pairs found. nodes_by_count={dict(nodes_by_count)}, target_count={target_count}"
        # )
        return None

    # Sort: cross-level first (priority 0), then same-level (priority 1), then fallback (priority 2)
    candidate_pairs.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
    best = candidate_pairs[0]
    return (best[4], best[5])


def _find_merging_pair_by_extremes(
    nodes_at_height: list[str],
    height: int,
    comp_count_lookup: dict[str, int],
    prefer_bottom: bool,
) -> tuple[str, str] | None:
    """Find merging pairs with preference for bottom or top nodes.

    IMPORTANT: Only match same extreme type - bottoms with bottoms, tops with tops.
    Never mix bottom with top.
    """
    # random.shuffle(nodes_at_height)
    node_set: set[str] = set(nodes_at_height)
    comp_to_nodes: dict[int, list[str]] = defaultdict(list)
    for n in nodes_at_height:
        comp_id = crystal_graph._chain._node_component[n]
        if comp_id is not None:
            comp_to_nodes[comp_id].append(n)

    comp_ids = list(comp_to_nodes.keys())
    if len(comp_ids) < 2:
        return None

    candidate_pairs: list[tuple[int, int, str, str]] = []

    def get_extremes_for_comp(cid):
        members = crystal_graph._chain._component_members[cid]
        tops = [
            n
            for n in members
            if n in node_set
            and not crystal_graph.get_node(n).get_links(better_than=True)
        ]
        bottoms = [
            n
            for n in members
            if n in node_set
            and not crystal_graph.get_node(n).get_links(worse_than=True)
        ]
        return tops, bottoms

    for i in range(len(comp_ids)):
        for j in range(i + 1, len(comp_ids)):
            comp_a, comp_b = comp_ids[i], comp_ids[j]

            tops_a, bottoms_a = get_extremes_for_comp(comp_a)
            tops_b, bottoms_b = get_extremes_for_comp(comp_b)

            if prefer_bottom:
                for a in bottoms_a:
                    for b in bottoms_b:
                        count_a, count_b = comp_count_lookup[a], comp_count_lookup[b]
                        candidate_pairs.append(
                            (count_a + count_b, abs(count_a - count_b), a, b)
                        )
                for a in tops_a:
                    for b in tops_b:
                        count_a, count_b = comp_count_lookup[a], comp_count_lookup[b]
                        candidate_pairs.append(
                            (count_a + count_b, abs(count_a - count_b), a, b)
                        )
            else:
                for a in tops_a:
                    for b in tops_b:
                        count_a, count_b = comp_count_lookup[a], comp_count_lookup[b]
                        candidate_pairs.append(
                            (count_a + count_b, abs(count_a - count_b), a, b)
                        )
                for a in bottoms_a:
                    for b in bottoms_b:
                        count_a, count_b = comp_count_lookup[a], comp_count_lookup[b]
                        candidate_pairs.append(
                            (count_a + count_b, abs(count_a - count_b), a, b)
                        )

    if not candidate_pairs:
        return None

    candidate_pairs.sort()
    best = candidate_pairs[0]
    return (best[2], best[3])


def _find_refinement_pair_in_component(
    nodes_at_comp_and_chain: list[str],
    comp_count_lookup: dict[str, int],
) -> tuple[str, str] | None:
    """Find a same-component refinement pair that is not already handled by collapsible logic."""
    component_groups: dict[int, list[str]] = defaultdict(list)
    for node in nodes_at_comp_and_chain:
        comp_id = crystal_graph._chain._node_component[node]
        if comp_id is not None:
            component_groups[comp_id].append(node)

    candidate_pairs: list[tuple[int, int, int, float, str, str]] = []

    for comp_id, members in component_groups.items():
        if len(members) < 2:
            continue

        # Prefer the least-compared nodes first; loop 2 should refine the current frontier
        # without scanning the whole component quadratically.
        sorted_members = heapq.nsmallest(
            48,
            members,
            key=lambda n: (
                comp_count_lookup[n],
                crystal_graph._chain._chain_length[n],
                float(crystal_graph._images[n]["score"]),
            ),
        )

        for i in range(len(sorted_members)):
            a = sorted_members[i]
            for j in range(i + 1, len(sorted_members)):
                b = sorted_members[j]

                if crystal_graph.are_in_same_path(a, b):
                    continue

                both_top = (
                    not crystal_graph._chain._better_than[a]
                    and not crystal_graph._chain._better_than[b]
                )
                both_bottom = (
                    not crystal_graph._chain._worse_than[a]
                    and not crystal_graph._chain._worse_than[b]
                )
                if both_top or both_bottom:
                    continue

                count_a = comp_count_lookup[a]
                count_b = comp_count_lookup[b]
                height_a = crystal_graph._chain._chain_length[a]
                height_b = crystal_graph._chain._chain_length[b]
                score_a = float(crystal_graph._images[a]["score"])
                score_b = float(crystal_graph._images[b]["score"])

                candidate_pairs.append(
                    (
                        max(count_a, count_b),
                        abs(count_a - count_b),
                        abs(height_a - height_b),
                        abs(score_a - score_b),
                        a,
                        b,
                    )
                )

    if not candidate_pairs:
        return None

    candidate_pairs.sort()
    best = candidate_pairs[0]
    return (best[4], best[5])


def _find_cross_component_progression_pair(
    target_nodes: list[str],
    candidate_filenames: set[str],
    comp_count_lookup: dict[str, int],
    target_count: int,
) -> tuple[str, str] | None:
    """Pair a lone active-level extreme with the nearest higher-count same-extreme node from another component."""
    if not target_nodes:
        return None

    candidate_pairs: list[tuple[int, int, int, int, str, str]] = []

    for node in target_nodes:
        node_comp = crystal_graph._chain._node_component[node]
        node_is_top = not crystal_graph._chain._better_than[node]
        node_is_bottom = not crystal_graph._chain._worse_than[node]
        node_height = crystal_graph._chain._chain_length[node]

        for partner in candidate_filenames:
            if partner == node:
                continue

            partner_comp = crystal_graph._chain._node_component[partner]
            if partner_comp == node_comp:
                continue

            partner_count = comp_count_lookup[partner]
            if partner_count <= target_count:
                continue

            partner_is_top = not crystal_graph._chain._better_than[partner]
            partner_is_bottom = not crystal_graph._chain._worse_than[partner]
            same_extreme = (node_is_top and partner_is_top) or (
                node_is_bottom and partner_is_bottom
            )
            if not same_extreme:
                continue

            if crystal_graph.are_in_same_path(node, partner):
                continue

            candidate_pairs.append(
                (
                    partner_count,
                    abs(partner_count - target_count),
                    abs(crystal_graph._chain._chain_length[partner] - node_height),
                    len(crystal_graph._chain._component_members[partner_comp]),
                    node,
                    partner,
                )
            )

    if not candidate_pairs:
        return None

    candidate_pairs.sort()
    best = candidate_pairs[0]
    return (best[4], best[5])


# def _find_unconnected_pair(nodes_at_height: list[str]) -> tuple[str, str] | None:
#     """Find a pair of nodes that are not in the same path."""
#     n = len(nodes_at_height)
#     for i in range(n):
#         for j in range(i + 1, n):
#             a, b = nodes_at_height[i], nodes_at_height[j]
#             if not crystal_graph.are_in_same_path(a, b):
#                 return (a, b)
#     return None


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

    # Invalidate images cache so next call gets fresh comparison counts
    global _images_cache
    _images_cache = {"data": None, "timestamp": 0.0}

    return True
