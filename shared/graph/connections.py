from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Any


@dataclass
class ComparisonConnections:
    all_images: list[dict[str, Any]]
    all_comparisons: list[dict[str, Any]]
    graph_comparisons: list[dict[str, Any]]
    compared_pairs: set[tuple[str, str]]
    winners_by_image: defaultdict[str, list[tuple[str, float]]]
    losers_by_image: defaultdict[str, list[tuple[str, float]]]
    winner_to_losers: defaultdict[str, list[str]]
    graph_nodes: set[str]
    all_filenames: set[str]
    undirected_graph: defaultdict[str, set[str]]
    component_by_filename: dict[str, int]
    chain_length_by_filename: dict[str, int]
    chain_members_by_component: dict[int, list[str]]


def _load_all_images() -> list[dict[str, Any]]:
    from external_modules.step01ranking_new.database.images_table import get_all_images

    return get_all_images()


def _load_all_comparisons() -> list[dict[str, Any]]:
    from external_modules.step01ranking_new.database.comparisons_table import (
        get_all_comparisons,
    )

    return get_all_comparisons()


def get_current_connections(
    all_images: list[dict[str, Any]] | None = None,
    all_comparisons: list[dict[str, Any]] | None = None,
    include_transitive: bool = False,
    comparison_weight: float | None = None,
) -> ComparisonConnections:
    if all_images is None:
        all_images = _load_all_images()
    if all_comparisons is None:
        all_comparisons = _load_all_comparisons()

    compared_pairs: set[tuple[str, str]] = set()
    winners_by_image = defaultdict(list)
    losers_by_image = defaultdict(list)
    winner_to_losers = defaultdict(list)
    winner_to_losers_seen: defaultdict[str, set[str]] = defaultdict(set)

    all_filenames = {
        str(image["filename"])
        for image in all_images
        if isinstance(image, dict) and image.get("filename")
    }
    graph_nodes: set[str] = set()
    undirected_graph = defaultdict(set)

    for filename in all_filenames:
        undirected_graph[filename]

    graph_comparisons: list[dict[str, Any]] = []
    for comp in all_comparisons:
        filename_a = comp.get("filename_a")
        filename_b = comp.get("filename_b")
        winner = comp.get("winner")

        if not filename_a or not filename_b or not winner:
            continue

        filename_a = str(filename_a)
        filename_b = str(filename_b)
        winner = str(winner)
        sorted_pair = sorted((filename_a, filename_b))
        compared_pairs.add((sorted_pair[0], sorted_pair[1]))
        all_filenames.add(filename_a)
        all_filenames.add(filename_b)

        transitive_depth = int(comp.get("transitive_depth", 0) or 0)
        if not include_transitive and transitive_depth > 0:
            continue

        weight = float(comp.get("weight", 1.0) or 1.0)
        if comparison_weight is not None and weight != comparison_weight:
            continue

        loser = filename_b if winner == filename_a else filename_a
        graph_comparisons.append(comp)
        graph_nodes.add(filename_a)
        graph_nodes.add(filename_b)

        winners_by_image[winner].append((loser, weight))
        losers_by_image[loser].append((winner, weight))
        if loser not in winner_to_losers_seen[winner]:
            winner_to_losers[winner].append(loser)
            winner_to_losers_seen[winner].add(loser)

        undirected_graph[filename_a].add(filename_b)
        undirected_graph[filename_b].add(filename_a)

    for filename in all_filenames:
        undirected_graph[filename]

    component_by_filename: dict[str, int] = {}
    chain_length_by_filename: dict[str, int] = {}
    chain_members_by_component: dict[int, list[str]] = {}
    visited: set[str] = set()
    visit_order = [
        str(image["filename"])
        for image in all_images
        if isinstance(image, dict) and image.get("filename")
    ]
    seen_in_order = set(visit_order)
    visit_order.extend(
        filename for filename in all_filenames if filename not in seen_in_order
    )
    component_id = 0

    for filename in visit_order:
        if filename in visited:
            continue

        queue = deque([filename])
        visited.add(filename)
        members: list[str] = []

        while queue:
            current = queue.popleft()
            members.append(current)

            for neighbor in undirected_graph[current]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                queue.append(neighbor)

        chain_members_by_component[component_id] = members
        chain_length = max(0, len(members) - 1)
        for member in members:
            component_by_filename[member] = component_id
            chain_length_by_filename[member] = chain_length

        component_id += 1

    return ComparisonConnections(
        all_images=all_images,
        all_comparisons=all_comparisons,
        graph_comparisons=graph_comparisons,
        compared_pairs=compared_pairs,
        winners_by_image=winners_by_image,
        losers_by_image=losers_by_image,
        winner_to_losers=winner_to_losers,
        graph_nodes=graph_nodes,
        all_filenames=all_filenames,
        undirected_graph=undirected_graph,
        component_by_filename=component_by_filename,
        chain_length_by_filename=chain_length_by_filename,
        chain_members_by_component=chain_members_by_component,
    )


def enumerate_simple_chains(
    connections: ComparisonConnections,
    max_chain_depth: int = 50,
    max_chains: int = 200000,
) -> tuple[list[list[str]], Counter[int]]:
    chains: list[list[str]] = []
    chain_lengths: Counter[int] = Counter()
    stack: list[tuple[str, list[str]]] = []

    for start in list(connections.graph_nodes):
        stack.append((start, [start]))
        while stack and len(chains) < max_chains:
            node, path = stack.pop()
            for neighbor in connections.winner_to_losers.get(node, []):
                if neighbor in path:
                    continue

                new_path = path + [neighbor]
                if len(new_path) >= 2:
                    chains.append(new_path)
                    chain_lengths[len(new_path)] += 1

                if len(new_path) < max_chain_depth:
                    stack.append((neighbor, new_path))

            if len(chains) >= max_chains:
                break

    return chains, chain_lengths
