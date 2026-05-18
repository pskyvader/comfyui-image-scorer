"""Cross-component merging and progression pair finders.

These strategies connect nodes from *different* components to merge
separate ranking hierarchies together.
"""

from typing import Any
from collections import defaultdict
import heapq
import logging

from shared.graph import crystal_graph
from .graph_helpers import is_top_node, is_bottom_node, get_node_component, get_chain_length

logger = logging.getLogger(__name__)


def find_merging_pair_by_extremes(
    nodes_at_height: list[str],
    height: int,
    comp_count_lookup: dict[str, int],
    prefer_bottom: bool,
) -> tuple[str, str] | None:
    """Find merging pairs with preference for bottom or top nodes.

    IMPORTANT: Only match same extreme type - bottoms with bottoms, tops with tops.
    Never mix bottom with top.
    """
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


def find_cross_component_progression_pair(
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
        node_comp = get_node_component(node)
        node_is_top = is_top_node(node)
        node_is_bottom = is_bottom_node(node)
        node_height = get_chain_length(node)

        for partner in candidate_filenames:
            if partner == node:
                continue

            partner_comp = get_node_component(partner)
            if partner_comp == node_comp:
                continue

            partner_count = comp_count_lookup[partner]
            if partner_count <= target_count:
                continue

            partner_is_top = is_top_node(partner)
            partner_is_bottom = is_bottom_node(partner)
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
                    abs(get_chain_length(partner) - node_height),
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


def find_refinement_pair_in_component(
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
