"""Read-only serialization of graph objects into comparison-frontend payloads.

Both functions accept ``NodeProxy`` objects directly so that, when the phase
selectors later return proxies instead of filename strings, no signature change
is required here. All field reading is isolated in the two helpers below.
"""

from __future__ import annotations

from typing import Any

from ....shared.config import config
from ....shared.graph.crystal_graph import crystal_graph
from ....shared.graph.node_proxy import NodeProxy

from .trueskill_rating import expected_win_probability, Rating


def _describe_one(node: NodeProxy) -> dict[str, Any]:
    """Build the per-image payload from a NodeProxy.

    Every field is read through the proxy; no database row is touched here.
    """
    component = node.get_component()
    chain = node.get_chain(only_main=True)
    chain_length = chain[0].length if chain else 0
    chain_id = chain[0].id if chain else None
    chain_main_members = (
        crystal_graph.get_main_chain_member_count(chain_id)
        if chain_id is not None
        else 0
    )

    seed_target = int(config["ranking"]["seed_target_comparisons"])
    is_seed = node.comparison_count >= seed_target

    extremes = {"top": 0, "bottom": 0}
    if component is not None:
        extremes = {
            "top": sum(1 for member in component.nodes if member.is_top()),
            "bottom": sum(1 for member in component.nodes if member.is_bottom()),
        }

    return {
        "filename": node.filename,
        "score": round(float(node.score), 4),
        "rating_mu": round(float(node.mu_skill), 4),
        "rating_sigma": round(float(node.sigma_uncertainty), 4),
        "comparison_count": int(node.comparison_count),
        "chain_length": chain_length,
        "chain_id": chain_id,
        "chain_main_members": chain_main_members,
        "component_size": component.size if component is not None else 0,
        "component_id": component.id if component is not None else None,
        "is_top": node.is_top(),
        "is_bottom": node.is_bottom(),
        "is_seed": is_seed,
        "_extremes": extremes,
    }


def describe_image(node: NodeProxy) -> dict[str, Any]:
    """Return all per-image info for a single node, regardless of phase."""
    return _describe_one(node)


def describe_pair(
    node_a: NodeProxy,
    node_b: NodeProxy,
    phase_index: int,
) -> dict[str, Any]:
    """Return phase-specific pair context built from the two nodes and config.

    ``phase_index`` is an int (0=seed, 1=anchor, 2=collapsible, 3=chain_merge,
    4=refine, 5=fallback). The int->label mapping lives on the frontend.
    """
    from . import graph_helpers
    from .constants import MIN_CHAIN_THRESHOLD

    ranking_conf = config["ranking"]

    left = _describe_one(node_a)
    right = _describe_one(node_b)

    base_level = min(left["comparison_count"], right["comparison_count"])

    level_counts: dict[int, int] = {}
    all_nodes = crystal_graph.get_all_nodes()
    for n in all_nodes:
        c = int(n.comparison_count)
        level_counts[c] = level_counts.get(c, 0) + 1

    rating_a = Rating(mu_skill=node_a.mu_skill, sigma_uncertainty=node_a.sigma_uncertainty)
    rating_b = Rating(mu_skill=node_b.mu_skill, sigma_uncertainty=node_b.sigma_uncertainty)
    probability_a_beats_b = expected_win_probability(rating_a, rating_b)

    component_a = node_a.get_component()
    component_b = node_b.get_component()
    same_component = (
        component_a is not None
        and component_b is not None
        and component_a.id == component_b.id
    )

    graph_stats = crystal_graph.get_graph_stats()

    return {
        "phase": phase_index,
        "total_images": len(all_nodes),
        "total_comparisons": graph_stats["total_comparisons"],
        "total_chains": graph_stats["total_chains"],
        "target_chains": MIN_CHAIN_THRESHOLD,
        "level": level_counts,
        "base_level": base_level,
        "collapsible": graph_helpers.is_collapsable_pair(
            node_a.filename, node_b.filename
        ),
        "same_component": same_component,
        "seed_percentage": int(ranking_conf["seed_percentage"]),
        "seed_target_comparisons": int(ranking_conf["seed_target_comparisons"]),
        "insertion_target_comparisons": int(
            ranking_conf["insertion_target_comparisons"]
        ),
        "reserve_count": int(ranking_conf["reserve_count"]),
        "probability_a_beats_b": round(probability_a_beats_b, 4),
    }
