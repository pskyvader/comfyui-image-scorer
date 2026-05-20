"""Loop 2 — Chain refinement pair selection.

Selects the two shortest chains and finds a pair of nodes across them
at the largest gap between shared waypoints.
"""

from typing import Any
import logging
from collections.abc import Iterator
from time import time
import random

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

logger: logging.Logger = logging.getLogger(__name__)


# nodeTuple = tuple[str, bool]
nodeList = list[NodeProxy]
# chainTuple = tuple[int, nodeList]
chainList = list[chainTuple]

chainPair = tuple[chainTuple, chainTuple]
chainPairList = list[chainPair]

nodePair = tuple[NodeProxy, NodeProxy, int, int]


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
        candidate_filenames, limit=512
    )
    if not valid_chains:
        logger.warning("[REFINEMENT] No chain pairs available")
        return None

    # subchain_pairs: chainPairList = _create_subchains(valid_chains, limit=128)
    subchain_pair: tuple[nodePair, list[nodeTuple], list[nodeTuple]] | None = (
        _create_subchains(valid_chains, candidate_filenames)
    )

    if not subchain_pair:
        return None
    res: tuple[tuple[str, str], tuple[ChainProxy, ChainProxy], dict[str, Any]] = (
        _create_pair_metadata(
            subchain_pair[0],
            subchain_pair[1],
            subchain_pair[2],
            comp_count_lookup,
            score_lookup,
        )
    )

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
    logger.debug(
        f"[NEXT-PAIR] Loop=2 Sub=chain_refinement "
        f"ChainLengths=({chains[0].length},{chains[0].length}) "
        f"Pair=({pair_string[0]}, {pair_string[1]}) "
        f"LeftCompCount={metadata['node_0']['comp_count']} "
        f"RightCompCount={metadata['node_1']['comp_count']}"
    )
    logger.debug(f"[REFINEMENT] Found pair in {time() - start_timer:.2f} seconds")

    return pair_string


def _get_prioritized_chain_pairs(
    candidate_filenames: set[str], limit: int = 8
) -> Iterator[chainTuple]:
    """Yield the shortest prioritized main chains."""

    main_chain_map: dict[int, chainDict] = crystal_graph.get_chains_map()

    yielded = 0

    for _length, chains in main_chain_map.items():
        # logger.debug(
        #     f"[CHAIN-PAIR] Processing chains of length {_length} with {len(chains)} chains"
        # )

        chain_list: list[chainTuple] = [
            chain_tuple
            for chain_tuple in chains.values()
            if chain_tuple[0].id not in chain_lru_cache
        ]
        random.shuffle(
            chain_list
        )  # Randomize order of same-length chains to improve diversity across runs

        for chain_id, chain_tuple in chains.items():
            _chain, nodes = chain_tuple

            if chain_id in chain_lru_cache:
                continue
            if len(nodes) < 3:
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
    chain_list: Iterator[chainTuple], candidate_filenames: set[str], limit: int = 10
) -> tuple[nodePair, list[nodeTuple], list[nodeTuple]] | None:

    seen: list[chainTuple] = []
    pair_list: list[Any] = []
    for chain_b in chain_list:
        for chain_a in seen:
            pair: tuple[nodePair, list[nodeTuple], list[nodeTuple]] | None = (
                _extract_subchains(chain_a, chain_b, candidate_filenames)
            )
            if pair:
                pair_list.append(pair)
                if len(pair_list) >= limit:
                    # return longest subchain
                    pair_list.sort(
                        key=lambda x: max(len(x[1]), len(x[2])), reverse=True
                    )
                    return pair_list[0]

        seen.append(chain_b)

    return None


def _extract_subchains(
    nodes_a: chainTuple, nodes_b: chainTuple, candidate_filenames: set[str]
) -> tuple[nodePair, list[nodeTuple], list[nodeTuple]] | None:
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
        logger.debug(
            f"Chain A: {[n[0].filename for n in chain_nodes_a]}, "
            f"Chain B: {[n[0].filename for n in chain_nodes_b]}"
        )
        return None

    # subchains: chainPairList = []

    # Head gap
    # first_a, first_b = waypoints[0]

    # if first_a > 0 or first_b > 0:
    #     sub_a = (chain_proxy_a, chain_nodes_a[:first_a])
    #     sub_b = (chain_proxy_b, chain_nodes_b[:first_b])
    #     pair: nodePair | None = _get_node_pair(sub_a[1], sub_b[1], candidate_filenames)
    #     if pair:
    #         return (pair, sub_a[1], sub_b[1])
    #     # subchains.append((sub_a, sub_b))

    # Middle gaps
    waypoints_pairs = list(zip(waypoints, waypoints[1:]))
    random.shuffle(
        waypoints_pairs
    )  # Randomize order of gaps to improve diversity across runs

    for (a1, b1), (a2, b2) in waypoints_pairs:

        if a2 - a1 > 1 or b2 - b1 > 1:

            sub_a = (
                chain_proxy_a,
                chain_nodes_a[a1 + 1 : a2],
            )

            sub_b = (
                chain_proxy_b,
                chain_nodes_b[b1 + 1 : b2],
            )

            # check if subchain can be collapsable (with subchains smaller than main chain)
            diff_a = a2 - a1 + 1
            diff_b = b2 - b1 + 1
            pair = None
            if diff_a < len_a and diff_b < len_b:
                if random.random() < 0.5:
                    top = True
                else:
                    top = False

                pair: nodePair | None = _get_node_pair(
                    sub_a[1], sub_b[1], candidate_filenames, top=top, bottom=not top
                )
                if not pair:
                    pair: nodePair | None = _get_node_pair(
                        sub_a[1], sub_b[1], candidate_filenames, top=not top, bottom=top
                    )

            if not pair:
                pair: nodePair | None = _get_node_pair(
                    sub_a[1], sub_b[1], candidate_filenames
                )

            if pair:
                # logger.debug(
                #     f"[SUBCHAIN] Found gap between waypoints: "
                #     f"A({a1} -> {a2}, len={len(sub_a[1])}),{a2-a1+1}, "
                #     f"B({b1} -> {b2}, len={len(sub_b[1])}),{b2-b1+1},"
                #     f"Chain length: A={len(chain_nodes_a)}, B={len(chain_nodes_b)}"
                # )

                return (pair, sub_a[1], sub_b[1])
                # subchains.append((sub_a, sub_b))

    # # Tail gap
    # last_a, last_b = waypoints[-1]

    # if last_a < len_a - 1 or last_b < len_b - 1:

    #     sub_a = (
    #         chain_proxy_a,
    #         chain_nodes_a[last_a + 1 :],
    #     )

    #     sub_b = (
    #         chain_proxy_b,
    #         chain_nodes_b[last_b + 1 :],
    #     )
    #     pair: nodePair | None = _get_node_pair(sub_a[1], sub_b[1], candidate_filenames)
    #     if pair:
    #         return (pair, sub_a[1], sub_b[1])
    #     # subchains.append((sub_a, sub_b))

    return None
    # # Sort after filtering
    # subchains.sort(
    #     key=lambda x: max(len(x[0][1]), len(x[1][1])),
    #     reverse=True,
    # )

    # # logger.debug(f"[SUBCHAIN] {len(subchains)} subchain pairs after filtering")

    # return subchains


def _get_node_pair(
    subchain_a: list[nodeTuple],
    subchain_b: list[nodeTuple],
    candidate_filenames: set[str],
    top: bool = False,
    bottom: bool = False,
) -> None | tuple[NodeProxy, NodeProxy, int, int]:
    if bottom:
        center_idx_a: int = len(subchain_a) - 1
        center_idx_b: int = len(subchain_b) - 1
    elif top:
        center_idx_a: int = 0
        center_idx_b: int = 0
    else:
        center_idx_a: int = len(subchain_a) // 2
        center_idx_b: int = len(subchain_b) // 2

    if center_idx_a > len(subchain_a) or center_idx_b > len(subchain_b):
        logger.debug(
            f"[SUBCHAIN] Skipping subchain pair with empty subchain: "
            f"LenA={len(subchain_a)}, LenB={len(subchain_b)}"
        )
        return None
    node_a, is_main_a = subchain_a[center_idx_a]
    node_b, is_main_b = subchain_b[center_idx_b]

    if not is_main_a or not is_main_b:
        # logger.debug(
        #     f"[SUBCHAIN] Skipping pair with non-main node: "
        #     f"index_a={center_idx_a} (main={is_main_a}), index_b={center_idx_b} (main={is_main_b}), "
        #     f"{node_a.filename} (main={is_main_a}) vs {node_b.filename} (main={is_main_b})"
        # )
        return None

    if (
        node_a.filename not in candidate_filenames
        or node_b.filename not in candidate_filenames
    ):
        # logger.debug(
        #     f"[SUBCHAIN] Skipping pair with center node not in candidates: "
        #     f"{node_a.filename} vs {node_b.filename}"
        # )

        return None
    if node_a.filename in node_lru_cache or node_b.filename in node_lru_cache:
        # logger.debug(
        #     f"[SUBCHAIN] Skipping pair with center node in LRU cache: "
        #     f"{node_a.filename} vs {node_b.filename}"
        # )
        return None

    # Both center nodes are valid; check if they're in the same path
    if crystal_graph.are_in_same_path(node_a.filename, node_b.filename):
        logger.debug(
            f"[SUBCHAIN] Skipping pair in same path: {node_a.filename} vs {node_b.filename}"
        )
        return None
    return (node_a, node_b, center_idx_a, center_idx_b)


def _create_pair_metadata(
    pair: nodePair,
    subchain_a: list[nodeTuple],
    subchain_b: list[nodeTuple],
    comp_count_lookup: dict[str, int],
    score_lookup: dict[str, float],
) -> tuple[tuple[str, str], tuple[ChainProxy, ChainProxy], dict[str, Any]]:
    """Iterate over subchains, selecting the valid pair with the closest scores."""

    node_a, node_b, center_idx_a, center_idx_b = pair

    score_a = score_lookup.get(node_a.filename, -1)
    score_b = score_lookup.get(node_b.filename, -1)
    score_diff = abs(score_a - score_b)

    # # Check score similarity constraint
    # max_score_diff = 0.3  # Limit score gap to 30%
    # if score_diff > max_score_diff:
    #     logger.debug(
    #         f"[SUBCHAIN] Idx {current_idx}: Score diff too large "
    #         f"({score_diff:.4f} > 0.3), skipping"
    #     )
    #     continue
    ca: ChainProxy = node_a.get_chain()[0]
    cb: ChainProxy = node_b.get_chain()[0]

    # Both center nodes are valid and similar in score; build metadata and return
    metadata = {
        "node_0": {
            "filename": node_a.filename,
            "score": round(score_a, 4),
            "comp_count": comp_count_lookup.get(node_a.filename, 0),
            "chain_id": ca.id,
            "chain_len": ca.length,
            "pos_in_chain": ca.node_position(node_a.filename) + 1,
            "sub_chain_len": len(subchain_a),
            "pos_in_sub_chain": center_idx_a + 1,
            "sub_chain_nodes": [
                {
                    "filename": n[0].filename,
                    "score": round(score_lookup.get(n[0].filename, -1), 4),
                }
                for n in subchain_a
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
            "filename": node_b.filename,
            "score": round(score_b, 4),
            "comp_count": comp_count_lookup.get(node_b.filename, 0),
            "chain_id": cb.id,
            "chain_len": cb.length,
            "pos_in_chain": cb.node_position(node_b.filename) + 1,
            "sub_chain_len": len(subchain_b),
            "pos_in_sub_chain": center_idx_b + 1,
            "sub_chain_nodes": [
                {
                    "filename": n[0].filename,
                    "score": round(score_lookup.get(n[0].filename, -1), 4),
                }
                for n in subchain_b
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
        # "sub_chain_idx": i,
        # "total_sub_chains": len(subchains),
        "common_nodes": list(
            set(n.filename for n in ca.nodes) & set(n.filename for n in cb.nodes)
        ),
        "common_subchain_nodes": list(
            set(n[0].filename for n in subchain_a)
            & set(n[0].filename for n in subchain_b)
        ),
    }

    # logger.debug(f"metadata: {metadata}")

    # logger.info(
    #     f"[REFINEMENT-SELECTION] Selected Center Pair: "
    #     f"{center_node_a} vs {center_node_b} (Diff={score_diff:.4f})"
    #     f" in Subchain idx {i} "
    # )
    # chain_pair_progress[key] = current_idx + 1
    return (
        (node_a.filename, node_b.filename),
        (ca, cb),
        metadata,
    )

    logger.warning(f"[SUBCHAIN] No valid center pairs found in subchains ")
    return None
