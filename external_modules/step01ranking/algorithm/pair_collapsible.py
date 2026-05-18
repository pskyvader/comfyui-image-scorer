"""Collapsible pair finders.

Contains the pair-at-height helper AND the critical
``find_collapsable_pair_by_extremes`` function which is preserved
**verbatim** from the original merge_sort_ranker.py.
"""

from typing import Any
from collections import defaultdict
import logging

from shared.graph import crystal_graph
from .graph_helpers import group_nodes_by_extreme

logger = logging.getLogger(__name__)


def find_collapsable_pair_at_height(
    nodes_at_height: list[str], height: int, comp_count_lookup: dict[str, int]
) -> tuple[str, str] | None:
    """Find collapsable pairs at a given height (multiple top/bottom nodes in same component).

    Priority: pairs with the least total comparison count first.
    Both nodes should ideally have similar comparison counts.
    """
    # Group top and bottom nodes by component
    top_by_component, bottom_by_component = group_nodes_by_extreme(nodes_at_height)

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

        return None

    # Sort by min_count first (smaller counts get priority), then total_count, then count_diff
    candidate_pairs.sort(key=lambda x: (x[0], x[1], x[2]))

    best = candidate_pairs[0]

    return (best[3], best[4])


# ========================================================================================
# _find_collapsable_pair_by_extremes — PRESERVED VERBATIM from the original
# ========================================================================================

def find_collapsable_pair_by_extremes(
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
