"""Loop 2 — Chain refinement pair selection.

Selects the two shortest chains and finds a pair of nodes across them
at the largest gap between shared waypoints.
"""

from typing import Any
import logging
from collections.abc import Iterator
from time import time

from shared.graph.crystal_graph import (
    crystal_graph,
    ChainProxy,
    NodeProxy,
    nodeTuple,
    chainTuple,
    chainDict,
)
from .constants import PAIR_TYPE_REFINEMENT
from .state import (
    set_last_pair_metadata,
    chain_lru_cache,
    node_lru_cache,
    clear_old_cache,
)

logger = logging.getLogger(__name__)


# nodeTuple = tuple[str, bool]
nodeList = list[NodeProxy]
# chainTuple = tuple[int, nodeList]
chainList = list[chainTuple]

chainPair = tuple[chainTuple, chainTuple]
chainPairList = list[chainPair]

nodePair = tuple[nodeTuple, nodeTuple]


def find_refinement_pairs(
    candidate_filenames: set[str],
    comp_count_lookup: dict[str, int],
    score_lookup: dict[str, float],
) -> tuple[str, str] | None:
    """Loop 2: Chain refinement within a single graph component.
    This iterates over the shortest chains possible, and chooses images fron
    different chains, at the middle of the longest possible sub chain, provided
    that the nodes are valid. if not, then proceed to the next chain pair
    """

    start_timer = time()

    # valid_chains: list[chainTuple] = _get_prioritized_chain_pairs(
    #     candidate_filenames, limit=128
    # )
    valid_chains: Iterator[chainTuple] = _get_prioritized_chain_pairs(
        candidate_filenames, limit=128
    )
    if not valid_chains:
        logger.warning("[REFINEMENT] No chain pairs available")
        return None

    # subchain_pairs: chainPairList = _create_subchains(valid_chains, limit=128)
    subchain_pairs: Iterator[chainPair] = _create_subchains(valid_chains, limit=128)
    res: (
        tuple[tuple[str, str], tuple[ChainProxy, ChainProxy], dict[str, Any]] | None
    ) = _find_pair_in_subchains(
        subchain_pairs,
        candidate_filenames,
        comp_count_lookup,
        score_lookup,
    )

    if res:
        pair_string, chains, metadata = res
        # Successfully found a refinement pair
        chain_lru_cache.append(chains[0].id)
        chain_lru_cache.append(chains[1].id)

        node_lru_cache.append(pair_string[0])
        node_lru_cache.append(pair_string[1])
        clear_old_cache()
        set_last_pair_metadata(
            {
                "pair_type": PAIR_TYPE_REFINEMENT,
                "chain_a": tuple([node.filename for node in chains[0].nodes]),
                "chain_b": tuple([node.filename for node in chains[1].nodes]),
                "chain_len_a": chains[0].length,
                "chain_len_b": chains[1].length,
                "left_comp_count": metadata["node_0"]["comp_count"],
                "right_comp_count": metadata["node_1"]["comp_count"],
                "refinement_details": metadata,
            }
        )
        logger.info(
            f"[NEXT-PAIR] Loop=2 Sub=chain_refinement "
            f"ChainLengths=({chains[0].length},{chains[0].length}) "
            f"Pair=({pair_string[0]}, {pair_string[1]}) "
            f"LeftCompCount={metadata['node_0']['comp_count']} "
            f"RightCompCount={metadata['node_1']['comp_count']}"
        )
        logger.debug(f"[REFINEMENT] Found pair in {time() - start_timer:.2f} seconds")

        return pair_string
    logger.warning("[REFINEMENT] No valid pairs found in any chain pair")
    return None


def _get_prioritized_chain_pairs(
    candidate_filenames: set[str], limit: int = 8
) -> Iterator[chainTuple]:
    """Yield the shortest prioritized main chains."""

    main_chain_map: dict[int, chainDict] = crystal_graph.get_chains_map()

    yielded = 0

    for _length, chains in main_chain_map.items():
        for chain_id, chain_tuple in chains.items():
            _chain, nodes = chain_tuple

            if chain_id in chain_lru_cache:
                continue

            for node, is_main in nodes:
                if is_main and node.filename in candidate_filenames:
                    yield chain_tuple

                    yielded += 1
                    break

            # logger.debug(f"[CHAIN-PAIR] currently yielded {yielded} chains")
            if yielded >= limit:
                logger.debug(
                    f"[CHAIN-PAIR] Found {yielded} candidate chains for refinement"
                )
                return

    logger.debug(f"[CHAIN-PAIR] Found {yielded} candidate chains for refinement")


def _create_subchains(
    chain_list: Iterator[chainTuple], limit: int = 8
) -> Iterator[chainPair]:

    yielded = 0
    seen: list[chainTuple] = []
    for chain_b in chain_list:

        for chain_a in seen:

            for pair in _extract_subchains(chain_a, chain_b):
                yield pair

                yielded += 1

                if yielded >= limit:
                    logger.debug(f"[SUBCHAIN] Created {yielded} subchain pairs")
                    return

        seen.append(chain_b)

    logger.debug(f"[SUBCHAIN] Created {yielded} subchain pairs")


def _extract_subchains(
    nodes_a: chainTuple,
    nodes_b: chainTuple,
) -> chainPairList:
    """Exclusively get subchains (gaps) out of two chains.
    Sorted from longest to shortest subchains.
    """

    chain_proxy_a, chain_nodes_a = nodes_a
    chain_proxy_b, chain_nodes_b = nodes_b

    len_a = len(chain_nodes_a)
    len_b = len(chain_nodes_b)

    # Build filename -> index maps directly
    idx_a: dict[str, int] = {}
    idx_b: dict[str, int] = {}

    for i, node in enumerate(chain_nodes_a):
        idx_a[node[0].filename] = i

    for i, node in enumerate(chain_nodes_b):
        idx_b[node[0].filename] = i

    # Ordered waypoints following chain A order
    waypoints: list[tuple[int, int]] = [
        (idx_a[f], idx_b[f]) for f in idx_a if f in idx_b
    ]
    if not waypoints:
        logger.debug("[SUBCHAIN] No shared waypoints found between chains")
        return []

    subchains: chainPairList = []

    # Head gap
    first_a, first_b = waypoints[0]

    if first_a > 0 or first_b > 0:
        sub_a = (chain_proxy_a, chain_nodes_a[:first_a])
        sub_b = (chain_proxy_b, chain_nodes_b[:first_b])

        if _get_node_pair(sub_a[1], sub_b[1]):
            subchains.append((sub_a, sub_b))

    # Middle gaps
    for (a1, b1), (a2, b2) in zip(waypoints, waypoints[1:]):

        if a2 - a1 > 1 or b2 - b1 > 1:

            sub_a = (
                chain_proxy_a,
                chain_nodes_a[a1 + 1 : a2],
            )

            sub_b = (
                chain_proxy_b,
                chain_nodes_b[b1 + 1 : b2],
            )

            if _get_node_pair(sub_a[1], sub_b[1]):
                subchains.append((sub_a, sub_b))

    # Tail gap
    last_a, last_b = waypoints[-1]

    if last_a < len_a - 1 or last_b < len_b - 1:

        sub_a = (
            chain_proxy_a,
            chain_nodes_a[last_a + 1 :],
        )

        sub_b = (
            chain_proxy_b,
            chain_nodes_b[last_b + 1 :],
        )

        if _get_node_pair(sub_a[1], sub_b[1]):
            subchains.append((sub_a, sub_b))

    # Sort after filtering
    subchains.sort(
        key=lambda x: max(len(x[0][1]), len(x[1][1])),
        reverse=True,
    )

    # logger.debug(f"[SUBCHAIN] {len(subchains)} subchain pairs after filtering")

    return subchains


def _get_node_pair(
    subchain_a: list[nodeTuple], subchain_b: list[nodeTuple]
) -> tuple[nodeTuple, nodeTuple, int, int] | None:
    center_idx_a: int = len(subchain_a) // 2
    center_idx_b: int = len(subchain_b) // 2

    if center_idx_a > len(subchain_a) or center_idx_b > len(subchain_b):
        logger.debug(
            f"[SUBCHAIN] Skipping subchain pair with empty subchain: "
            f"LenA={len(subchain_a)}, LenB={len(subchain_b)}"
        )
        return None

    is_main_a: bool = subchain_a[center_idx_a][1]
    is_main_b: bool = subchain_b[center_idx_b][1]
    if is_main_a and is_main_b:
        return (
            subchain_a[center_idx_a],
            subchain_b[center_idx_b],
            center_idx_a,
            center_idx_b,
        )
    # logger.debug(
    #     f"[SUBCHAIN] Center nodes not both main (MainA={is_main_a}, MainB={is_main_b}), skipping"
    # )
    return None


def _find_pair_in_subchains(
    subchains: Iterator[chainPair],
    candidate_filenames: set[str],
    comp_count_lookup: dict[str, int],
    score_lookup: dict[str, float],
) -> tuple[tuple[str, str], tuple[ChainProxy, ChainProxy], dict[str, Any]] | None:
    """Iterate over subchains, selecting the valid pair with the closest scores."""
    if not subchains:
        logger.warning("[SUBCHAIN] No subchains to process")
        return None

    for i, (sub_a, sub_b) in enumerate(subchains):
        node_pair: tuple[nodeTuple, nodeTuple, int, int] | None = _get_node_pair(
            sub_a[1], sub_b[1]
        )

        if not node_pair:
            logger.debug(f"[SUBCHAIN] Idx {i}: Center node(s) not main, skipping")
            continue

        center_node_a: nodeTuple = node_pair[0]
        center_node_b: nodeTuple = node_pair[1]
        center_idx_a: int = node_pair[2]
        center_idx_b: int = node_pair[3]

        if (
            center_node_a[0].filename not in candidate_filenames
            or center_node_b[0].filename not in candidate_filenames
        ):
            # logger.debug(
            #     f"[SUBCHAIN] Idx {current_idx}: Node A not in candidates, skipping"
            # )
            continue
        if (
            center_node_a[0].filename in node_lru_cache
            or center_node_b[0].filename in node_lru_cache
        ):
            # logger.debug(f"[SUBCHAIN] Idx {current_idx}: Node A in LRU cache, skipping")
            continue

        # Both center nodes are valid; check if they're in the same path
        if crystal_graph.are_in_same_path(
            center_node_a[0].filename, center_node_b[0].filename
        ):
            # logger.debug(
            #     f"[SUBCHAIN] Idx {current_idx}: Both nodes in same path, skipping"
            # )
            continue

        score_a = score_lookup.get(center_node_a[0].filename, -1)
        score_b = score_lookup.get(center_node_b[0].filename, -1)
        score_diff = abs(score_a - score_b)

        # # Check score similarity constraint
        # max_score_diff = 0.3  # Limit score gap to 30%
        # if score_diff > max_score_diff:
        #     logger.debug(
        #         f"[SUBCHAIN] Idx {current_idx}: Score diff too large "
        #         f"({score_diff:.4f} > 0.3), skipping"
        #     )
        #     continue
        ca: ChainProxy = sub_a[0]
        cb: ChainProxy = sub_b[0]

        # Both center nodes are valid and similar in score; build metadata and return
        metadata = {
            "node_0": {
                "filename": center_node_a[0].filename,
                "score": round(score_a, 4),
                "comp_count": comp_count_lookup.get(center_node_a[0].filename, 0),
                "chain_id": ca.id,
                "chain_len": ca.length,
                "pos_in_chain": ca.node_position(center_node_a[0].filename) + 1,
                "sub_chain_len": len(sub_a[1]),
                "pos_in_sub_chain": center_idx_a + 1,
                "sub_chain_nodes": [
                    {
                        "filename": n[0].filename,
                        "score": round(score_lookup.get(n[0].filename, -1), 4),
                    }
                    for n in sub_a[1]
                ],
                "full_chain_nodes": [
                    {
                        "filename": n.filename,
                        "score": round(score_lookup.get(n.filename, -1), 4),
                    }
                    for n in ca.nodes
                ],
            },
            "node_1": {
                "filename": center_node_b[0].filename,
                "score": round(score_b, 4),
                "comp_count": comp_count_lookup.get(center_node_b[0].filename, 0),
                "chain_id": cb.id,
                "chain_len": cb.length,
                "pos_in_chain": cb.node_position(center_node_b[0].filename) + 1,
                "sub_chain_len": len(sub_b[1]),
                "pos_in_sub_chain": center_idx_b + 1,
                "sub_chain_nodes": [
                    {
                        "filename": n[0].filename,
                        "score": round(score_lookup.get(n[0].filename, -1), 4),
                    }
                    for n in sub_b[1]
                ],
                "full_chain_nodes": [
                    {
                        "filename": n.filename,
                        "score": round(score_lookup.get(n.filename, -1), 4),
                    }
                    for n in cb.nodes
                ],
            },
            "score_diff": round(score_diff, 4),
            "sub_chain_idx": i,
            # "total_sub_chains": len(subchains),
            "common_nodes": list(
                set(n.filename for n in ca.nodes) & set(n.filename for n in cb.nodes)
            ),
            "common_subchain_nodes": list(
                set(n[0].filename for n in sub_a[1])
                & set(n[0].filename for n in sub_b[1])
            ),
        }

        # logger.debug(f"metadata: {metadata}")

        logger.info(
            f"[REFINEMENT-SELECTION] Selected Center Pair: "
            f"{center_node_a} vs {center_node_b} (Diff={score_diff:.4f})"
            f" in Subchain idx {i} "
        )
        # chain_pair_progress[key] = current_idx + 1
        return (
            (center_node_a[0].filename, center_node_b[0].filename),
            (ca, cb),
            metadata,
        )

    logger.warning(f"[SUBCHAIN] No valid center pairs found in subchains ")
    return None
