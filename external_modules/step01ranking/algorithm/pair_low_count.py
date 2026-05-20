"""Loop 1 — Low comparison-count pair selection.

The big multi-phase pipeline: Comparison count → Component size →
Chain height → Collapsible / Cross-level / Merge / Progression.
"""

import logging
from dataclasses import dataclass
from typing import TypedDict

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
    get_component_members,
    get_chain_length,
)
from .pair_collapsible import find_collapsable_pair_by_extremes
from .pair_merge import (
    find_merging_pair_by_extremes,
    find_cross_component_progression_pair,
)

logger = logging.getLogger(__name__)


@dataclass
class PairContext:
    """Immutable context for pair selection."""

    candidate_filenames: set[str]
    comp_count_lookup: dict[str, int]
    target_comparison_count: int
    target_component_size: int
    target_chain_height: int

    def comp_count(self, node: str) -> int:
        """Get comparison count for a node."""
        return self.comp_count_lookup.get(node, 0)

    def is_candidate(self, node: str) -> bool:
        """Check if node is in candidate set."""
        return node in self.candidate_filenames


class PairMetadata(TypedDict, total=False):
    """Metadata recorded for a selected pair."""

    pair_type: str
    chain_level: int
    component_size: int
    left_comp_count: int
    right_comp_count: int


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def record_and_return_pair(
    pair: tuple[str, str],
    pair_type: str,
    ctx: PairContext,
    metadata_overrides: dict | None = None,
) -> tuple[str, str]:
    """Record pair metadata, log it, and return the pair."""
    metadata: PairMetadata = {
        "pair_type": pair_type,
        "chain_level": ctx.target_chain_height,
        "component_size": ctx.target_component_size,
        "left_comp_count": ctx.comp_count(pair[0]),
        "right_comp_count": ctx.comp_count(pair[1]),
    }
    if metadata_overrides:
        metadata.update(metadata_overrides)

    set_last_pair_metadata(metadata)

    logger.info(
        f"[NEXT-PAIR] Loop=1 Sub={pair_type} "
        f"ComparisonCount={ctx.target_comparison_count} "
        f"ComponentSize={ctx.target_component_size} "
        f"ChainHeight={ctx.target_chain_height} "
        f"Pair={pair} "
        f"LeftComparisonCount={ctx.comp_count(pair[0])} "
        f"RightComparisonCount={ctx.comp_count(pair[1])}"
    )

    return pair


def filter_nodes_by_count(
    nodes: list[str],
    comp_count_lookup: dict[str, int],
    max_count: int,
) -> list[str]:
    """Return nodes with comparison count <= max_count."""
    return [n for n in nodes if comp_count_lookup.get(n, 0) <= max_count]


def filter_nodes_by_chain_height(
    nodes: list[str],
    max_height: int,
) -> list[str]:
    """Return nodes with chain_length <= max_height."""
    return [n for n in nodes if get_chain_length(n) <= max_height]


def get_extreme_nodes(
    nodes: list[str],
) -> list[str]:
    """Return nodes that are either top or bottom (have minimal connections)."""
    return [n for n in nodes if is_top_node(n) or is_bottom_node(n)]


def get_all_chain_heights() -> list[int]:
    """Get sorted list of all chain heights in the graph."""
    chain_data = crystal_graph._chain._nodes_by_length
    return sorted(chain_data.keys()) if chain_data else []


def get_all_component_sizes() -> list[int]:
    """Get sorted list of unique component sizes."""
    components = crystal_graph.get_all_components()
    sizes = {comp.size for comp in components}
    return sorted(sizes)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def find_low_count_pairs(
    candidate_filenames: set[str],
    comp_count_lookup: dict[str, int],
    only_wins: list[str],
    only_losses: list[str],
) -> tuple[str, str] | None:
    """Loop 1: Low comparison count - Comparison count -> Component size -> Chain height -> Collapsible/Merge."""
    # check if only 1 top and 1 bottom, for early exit

    if len(only_wins) == 1 and len(only_losses) == 1:
        return None
    comparison_counts: list[int] = sorted(
        c for c in set(comp_count_lookup.values()) if c > 0
    )

    # Multi-component path - always prioritize efficiency
    return multi_component_path(
        candidate_filenames, comp_count_lookup, comparison_counts
    )


# ---------------------------------------------------------------------------
# Multi-component path
# ---------------------------------------------------------------------------


def multi_component_path(
    candidate_filenames: set[str],
    comp_count_lookup: dict[str, int],
    comparison_counts: list[int],
) -> tuple[str, str] | None:
    """
    Handle the multi-component case with efficient prioritization.

    Loop order: comparison_count -> component_size -> COLLAPSIBLE (all chain heights)
                -> MERGE (all chain heights)

    This ensures collapsible pairs consolidate components before attempting
    cross-component merges, reducing node skipping.
    """
    sorted_component_sizes = get_all_component_sizes()
    chain_heights = get_all_chain_heights()

    # logger.debug(
    #     f"Multi-component path with {len(comparison_counts)} comparison counts, "
    #     f"{len(sorted_component_sizes)} component sizes, "
    #     f"{len(chain_heights)} chain heights"
    #     f"comparison count sizes: {comparison_counts}, component sizes: {sorted_component_sizes}, chain heights: {chain_heights}"
    # )

    for target_comparison_count in comparison_counts:
        # Get all extreme nodes (top/bottom) at this comparison count
        nodes_below_count = filter_nodes_by_count(
            list(candidate_filenames), comp_count_lookup, target_comparison_count
        )
        extreme_nodes = get_extreme_nodes(nodes_below_count)
        logger.debug(
            f"Comparison count {target_comparison_count}: {len(extreme_nodes)} extreme nodes out of {len(nodes_below_count)} candidates"
        )

        for target_component_size in sorted_component_sizes:
            # Get nodes within this component size threshold
            nodes_at_component = [
                n
                for n in extreme_nodes
                if (comp_id := get_node_component(n)) is not None
                and (comp := crystal_graph.get_component(node_id=comp_id)) is not None
                and comp.size <= target_component_size
            ]

            nodes_at_or_below_target = filter_nodes_by_count(
                nodes_at_component, comp_count_lookup, target_comparison_count
            )

            if not nodes_at_or_below_target:
                continue

            # PHASE A: Find all collapsible pairs first (all chain heights)
            for target_chain_height in chain_heights:
                ctx = PairContext(
                    candidate_filenames,
                    comp_count_lookup,
                    target_comparison_count,
                    target_component_size,
                    target_chain_height,
                )
                pair = phase_collapsible(
                    nodes_at_or_below_target,
                    ctx,
                    candidate_filenames,
                )
                if pair:
                    return pair

            # PHASE C: Find merge pairs (all chain heights)
            for target_chain_height in chain_heights:
                ctx = PairContext(
                    candidate_filenames,
                    comp_count_lookup,
                    target_comparison_count,
                    target_component_size,
                    target_chain_height,
                )
                pair = phase_merge(
                    nodes_at_or_below_target,
                    ctx,
                    candidate_filenames,
                )
                if pair:
                    return pair

    return None


# ---------------------------------------------------------------------------
# Phase helpers
# ---------------------------------------------------------------------------


def phase_collapsible(
    nodes: list[str],
    ctx: PairContext,
    candidate_filenames: set[str],
) -> tuple[str, str] | None:
    """PHASE A: Find collapsible pairs at current chain height."""
    if not nodes:
        return None

    nodes_at_chain = filter_nodes_by_chain_height(nodes, ctx.target_chain_height)
    if not nodes_at_chain:
        return None

    # Try bottom first, then top
    for prefer_bottom in (True, False):
        pair = find_collapsable_pair_by_extremes(
            nodes_at_chain,
            ctx.target_chain_height,
            ctx.comp_count_lookup,
            prefer_bottom=prefer_bottom,
            target_count=ctx.target_comparison_count,
            candidate_filenames=candidate_filenames,
        )
        if pair:
            return record_and_return_pair(
                pair, "collapsible_bottom" if prefer_bottom else "collapsible_top", ctx
            )

    return None


def phase_merge(
    nodes: list[str],
    ctx: PairContext,
    candidate_filenames: set[str],
) -> tuple[str, str] | None:
    """PHASE C: Merge pairs (different components at same level)."""
    nodes_at_chain = filter_nodes_by_chain_height(nodes, ctx.target_chain_height)
    if len(nodes_at_chain) < 2:
        return None

    # Try merge pairs with bottom preference first
    for prefer_bottom in (True, False):
        pair = find_merging_pair_by_extremes(
            nodes_at_chain,
            ctx.target_chain_height,
            ctx.comp_count_lookup,
            prefer_bottom=prefer_bottom,
        )
        if pair:
            return record_and_return_pair(
                pair,
                "merge_bottom" if prefer_bottom else "merge_top",
                ctx,
            )

    # Try cross-component progression
    exact_target_nodes = [
        n for n in nodes_at_chain if ctx.comp_count(n) == ctx.target_comparison_count
    ]

    pair = find_cross_component_progression_pair(
        exact_target_nodes,
        candidate_filenames,
        ctx.comp_count_lookup,
        ctx.target_comparison_count,
    )
    if pair:
        return record_and_return_pair(pair, "cross_component_progression", ctx)

    return None
