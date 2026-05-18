"""Loop 1 — Low comparison-count pair selection.

The big multi-phase pipeline: Comparison count → Component size →
Chain height → Collapsible / Cross-level / Merge / Progression.
"""

from typing import Any
import logging

from shared.graph import crystal_graph
from external_modules.step01ranking.database.comparisons_table import (
    get_images_with_only_wins,
    get_images_with_only_losses,
)
from .constants import PAIR_TYPE_COLLAPSIBLE, PAIR_TYPE_WORST_WITH_WORST
from .state import set_last_pair_metadata
from .graph_helpers import (
    is_top_node,
    is_bottom_node,
    get_node_component,
    get_chain_length,
)
from .pair_collapsible import find_collapsable_pair_by_extremes
from .pair_merge import (
    find_merging_pair_by_extremes,
    find_cross_component_progression_pair,
)

logger = logging.getLogger(__name__)


def find_low_count_pairs(
    candidate_filenames: set[str],
    comp_count_lookup: dict[str, int],
) -> tuple[str, str] | None:
    """Loop 1: Low comparison count - Comparison count -> Component size -> Chain height -> Collapsible/Merge."""
    # check if theres exactly one top and one bottom node, from database

    only_wins = get_images_with_only_wins()
    only_losses = get_images_with_only_losses()

    if len(only_wins) == 1 and len(only_losses) == 1:
        return None

    comparison_counts: list[int] = sorted(
        c for c in set(comp_count_lookup.values()) if c > 0
    )

    # --- Single-component fast path ---
    if len(crystal_graph.get_all_components()) <= 1:
        return _single_component_path(
            candidate_filenames, comp_count_lookup, comparison_counts
        )

    # --- Multi-component path ---
    return _multi_component_path(
        candidate_filenames, comp_count_lookup, comparison_counts
    )


# ---------------------------------------------------------------------------
# Single-component fast path
# ---------------------------------------------------------------------------


def _single_component_path(
    candidate_filenames: set[str],
    comp_count_lookup: dict[str, int],
    comparison_counts: list[int],
) -> tuple[str, str] | None:
    """Handle the single-component case with a simplified loop."""
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
            n for n in extreme_nodes if comp_count_lookup[n] <= target_comparison_count
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
                pair = find_collapsable_pair_by_extremes(
                    nodes_at_chain,
                    -1,
                    comp_count_lookup,
                    prefer_bottom=prefer_bottom,
                    target_count=target_comparison_count,
                    candidate_filenames=candidate_filenames,
                )
                if not pair:
                    continue

                set_last_pair_metadata(
                    {
                        "pair_type": PAIR_TYPE_COLLAPSIBLE,
                        "chain_level": target_chain_height,
                        "component_size": component_size,
                        "left_comp_count": comp_count_lookup[pair[0]],
                        "right_comp_count": comp_count_lookup[pair[1]],
                    }
                )
                logger.info(
                    f"[NEXT-PAIR] Loop=1 Sub=single_component_collapsible "
                    f"ComparisonCount={target_comparison_count} "
                    f"ChainHeight={target_chain_height} Pair={pair} "
                    f"LeftComparisonCount={comp_count_lookup[pair[0]]} "
                    f"RightComparisonCount={comp_count_lookup[pair[1]]}"
                )
                return pair
    return None


# ---------------------------------------------------------------------------
# Multi-component path
# ---------------------------------------------------------------------------


def _multi_component_path(
    candidate_filenames: set[str],
    comp_count_lookup: dict[str, int],
    comparison_counts: list[int],
) -> tuple[str, str] | None:
    """Handle the multi-component case with the full 3-phase pipeline."""
    component_sizes: dict[int, int] = {
        comp.id: comp.size for comp in crystal_graph.get_all_components()
    }
    sorted_component_sizes: list[int] = sorted(set(component_sizes.values()))
    chain_heights: list[int] = sorted(crystal_graph._chain._nodes_by_length.keys())

    for target_comparison_count in comparison_counts:
        # Step 1: Get all nodes with count <= target_count
        nodes_up_to_count = [
            n
            for n in candidate_filenames
            if comp_count_lookup[n] <= target_comparison_count
        ]

        # Step 2: Filter to only extremes (top/bottom nodes only)
        top_nodes: list[str] = [n for n in nodes_up_to_count if is_top_node(n)]
        bottom_nodes: list[str] = [n for n in nodes_up_to_count if is_bottom_node(n)]
        if len(top_nodes) > 0 or len(bottom_nodes) > 0:
            pass
        extreme_nodes = top_nodes + bottom_nodes

        # Step 3: Loop count -> component -> chain -> pair type
        for target_component_size in sorted_component_sizes:
            nodes_at_component = [
                n
                for n in extreme_nodes
                if crystal_graph.get_component(node_id=n).id <= target_component_size
            ]
            if len(nodes_at_component) > 0:
                pass
            # Separate nodes at or below target count from nodes with higher count
            nodes_at_or_below_target = [
                n
                for n in nodes_at_component
                if comp_count_lookup[n] <= target_comparison_count
            ]

            # PHASE A: Same-level collapsible pairs
            if len(nodes_at_or_below_target) > 0:
                pass
            pair = _phase_a_collapsible(
                nodes_at_or_below_target,
                chain_heights,
                comp_count_lookup,
                target_comparison_count,
                target_component_size,
                candidate_filenames,
            )
            if pair:
                return pair

            # PHASE B: Cross-level pairs
            pair = _phase_b_cross_level(
                nodes_at_component,
                nodes_at_or_below_target,
                candidate_filenames,
                comp_count_lookup,
                target_comparison_count,
                target_component_size,
            )
            if pair:
                return pair

            # PHASE C: Same-level merge pairs (different components)
            pair = _phase_c_merge(
                nodes_at_or_below_target,
                chain_heights,
                comp_count_lookup,
                candidate_filenames,
                target_comparison_count,
                target_component_size,
            )
            if pair:
                return pair

    return None


# ---------------------------------------------------------------------------
# Phase helpers
# ---------------------------------------------------------------------------


def _phase_a_collapsible(
    nodes_at_or_below_target: list[str],
    chain_heights: list[int],
    comp_count_lookup: dict[str, int],
    target_comparison_count: int,
    target_component_size: int,
    candidate_filenames: set[str],
) -> tuple[str, str] | None:
    """PHASE A: Same-level collapsible pairs (nodes at or below target count, same extreme)."""
    if len(nodes_at_or_below_target) < 1:
        return None

    for target_chain_height in chain_heights:
        nodes_at_chain = [
            n
            for n in nodes_at_or_below_target
            if crystal_graph._chain._chain_length[n] <= target_chain_height
        ]

        # Try collapsible bottom first
        pair = find_collapsable_pair_by_extremes(
            nodes_at_chain,
            -1,
            comp_count_lookup,
            prefer_bottom=True,
            target_count=target_comparison_count,
            candidate_filenames=candidate_filenames,
        )
        if pair:
            comp_a = crystal_graph._chain._node_component[pair[0]]
            comp_b = crystal_graph._chain._node_component[pair[1]]
            same_comp = "same" if comp_a == comp_b else "diff"
            set_last_pair_metadata(
                {
                    "pair_type": PAIR_TYPE_COLLAPSIBLE,
                    "chain_level": target_chain_height,
                    "component_size": target_component_size,
                    "left_comp_count": comp_count_lookup[pair[0]],
                    "right_comp_count": comp_count_lookup[pair[1]],
                }
            )
            logger.info(
                f"[NEXT-PAIR] Loop=1 Sub=collapsible_bottom ComparisonCount={target_comparison_count} ComponentSize={target_component_size} ChainHeight={target_chain_height} Pair={pair} SameComponent={same_comp} LeftComparisonCount={comp_count_lookup[pair[0]]} RightComparisonCount={comp_count_lookup[pair[1]]}"
            )
            return pair

        # Try collapsible top
        pair = find_collapsable_pair_by_extremes(
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
            set_last_pair_metadata(
                {
                    "pair_type": PAIR_TYPE_COLLAPSIBLE,
                    "chain_level": target_chain_height,
                    "component_size": target_component_size,
                    "left_comp_count": comp_count_lookup[pair[0]],
                    "right_comp_count": comp_count_lookup[pair[1]],
                }
            )
            logger.info(
                f"[NEXT-PAIR] Loop=1 Sub=collapsible_top ComparisonCount={target_comparison_count} ComponentSize={target_component_size} ChainHeight={target_chain_height} Pair={pair} SameComponent={same_comp} LeftComparisonCount={comp_count_lookup[pair[0]]} RightComparisonCount={comp_count_lookup[pair[1]]}"
            )
            return pair

    return None


def _phase_b_cross_level(
    nodes_at_component: list[str],
    nodes_at_or_below_target: list[str],
    candidate_filenames: set[str],
    comp_count_lookup: dict[str, int],
    target_comparison_count: int,
    target_component_size: int,
) -> tuple[str, str] | None:
    """PHASE B: Cross-level pairs - low count extreme paired with higher count node in same component."""
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
    if not (nodes_at_or_below_target and higher_nodes):
        return None

    for node in nodes_at_or_below_target:
        node_comp = crystal_graph._chain._node_component[node]
        node_is_bottom = is_bottom_node(node)
        node_is_top = is_top_node(node)
        if not node_is_bottom and not node_is_top:
            continue
        node_chain = get_chain_length(node)
        all_component_nodes = crystal_graph._chain._component_members[node_comp]
        partner_candidates = [
            n
            for n in all_component_nodes
            if n != node
            and n in candidate_filenames
            and comp_count_lookup[n] > target_comparison_count
            and not crystal_graph.are_in_same_path(node, n)
        ]
        for partner in partner_candidates:
            partner_is_bottom = is_bottom_node(partner)
            partner_is_top = is_top_node(partner)
            same_extreme = (node_is_bottom and partner_is_bottom) or (
                node_is_top and partner_is_top
            )
            if same_extreme:
                pair = (node, partner)
                set_last_pair_metadata(
                    {
                        "pair_type": "cross_level_pair",
                        "chain_level": node_chain,
                        "component_size": target_component_size,
                        "left_comp_count": comp_count_lookup[pair[0]],
                        "right_comp_count": comp_count_lookup[pair[1]],
                    }
                )
                logger.info(
                    f"[NEXT-PAIR] Loop=1 Sub=cross_level_pair ComparisonCount={target_comparison_count} ComponentSize={target_component_size} ChainHeight={node_chain} Pair={pair} LeftComparisonCount={comp_count_lookup[pair[0]]} RightComparisonCount={comp_count_lookup[pair[1]]}"
                )
                return pair

    return None


def _phase_c_merge(
    nodes_at_or_below_target: list[str],
    chain_heights: list[int],
    comp_count_lookup: dict[str, int],
    candidate_filenames: set[str],
    target_comparison_count: int,
    target_component_size: int,
) -> tuple[str, str] | None:
    """PHASE C: Same-level merge pairs (different components)."""
    for target_chain_height in chain_heights:
        nodes_at_chain = [
            n
            for n in nodes_at_or_below_target
            if crystal_graph._chain._chain_length[n] <= target_chain_height
        ]
        if not nodes_at_chain:
            continue

        if len(nodes_at_chain) >= 2:
            pair = find_merging_pair_by_extremes(
                nodes_at_chain,
                target_chain_height,
                comp_count_lookup,
                prefer_bottom=True,
            )
            if pair:
                comp_a = crystal_graph._chain._node_component[pair[0]]
                comp_b = crystal_graph._chain._node_component[pair[1]]
                set_last_pair_metadata(
                    {
                        "pair_type": PAIR_TYPE_WORST_WITH_WORST,
                        "chain_level": target_chain_height,
                        "component_size": target_component_size,
                        "left_comp_count": comp_count_lookup[pair[0]],
                        "right_comp_count": comp_count_lookup[pair[1]],
                    }
                )
                logger.info(
                    f"[NEXT-PAIR] Loop=1 Sub=merge_bottom ComparisonCount={target_comparison_count} ComponentSize={target_component_size} ChainHeight={target_chain_height} Pair={pair} ComponentA={comp_a} ComponentB={comp_b} LeftComparisonCount={comp_count_lookup[pair[0]]} RightComparisonCount={comp_count_lookup[pair[1]]}"
                )
                return pair

            pair = find_merging_pair_by_extremes(
                nodes_at_chain,
                target_chain_height,
                comp_count_lookup,
                prefer_bottom=False,
            )
            if pair:
                comp_a = crystal_graph._chain._node_component[pair[0]]
                comp_b = crystal_graph._chain._node_component[pair[1]]
                set_last_pair_metadata(
                    {
                        "pair_type": PAIR_TYPE_WORST_WITH_WORST,
                        "chain_level": target_chain_height,
                        "component_size": target_component_size,
                        "left_comp_count": comp_count_lookup[pair[0]],
                        "right_comp_count": comp_count_lookup[pair[1]],
                    }
                )
                logger.info(
                    f"[NEXT-PAIR] Loop=1 Sub=merge_top ComparisonCount={target_comparison_count} ComponentSize={target_component_size} ChainHeight={target_chain_height} Pair={pair} ComponentA={comp_a} ComponentB={comp_b} LeftComparisonCount={comp_count_lookup[pair[0]]} RightComparisonCount={comp_count_lookup[pair[1]]}"
                )
                return pair

        exact_target_nodes = [
            n for n in nodes_at_chain if comp_count_lookup[n] == target_comparison_count
        ]
        pair = find_cross_component_progression_pair(
            exact_target_nodes,
            candidate_filenames,
            comp_count_lookup,
            target_comparison_count,
        )
        if pair:
            comp_a = crystal_graph._chain._node_component[pair[0]]
            comp_b = crystal_graph._chain._node_component[pair[1]]
            set_last_pair_metadata(
                {
                    "pair_type": "cross_component_progression",
                    "chain_level": target_chain_height,
                    "component_size": target_component_size,
                    "left_comp_count": comp_count_lookup[pair[0]],
                    "right_comp_count": comp_count_lookup[pair[1]],
                }
            )
            logger.info(
                f"[NEXT-PAIR] Loop=1 Sub=cross_component_progression ComparisonCount={target_comparison_count} ComponentSize={target_component_size} ChainHeight={target_chain_height} Pair={pair} ComponentA={comp_a} ComponentB={comp_b} LeftComparisonCount={comp_count_lookup[pair[0]]} RightComparisonCount={comp_count_lookup[pair[1]]}"
            )
            return pair
    return None
