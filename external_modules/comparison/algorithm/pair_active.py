"""Active pair selection for the TrueSkill-based step01 flow."""

from __future__ import annotations

from collections import defaultdict
import time
from typing import Any, Iterator
import random

from ....shared.graph.chain_proxy import ChainProxy

from ....shared.graph.crystal_graph import NodeProxy, crystal_graph, NodeTuple

from ....shared.config import config
from ....shared.logger import SharedLogger
from ...database_structure.comparisons_table import (
    get_all_comparisons,
    get_images_with_only_wins,
    get_images_with_only_losses,
)

from .constants import MIN_CHAIN_THRESHOLD
from .trueskill_rating import expected_win_probability, rating_from_row

logger = SharedLogger.get_logger(__name__)


def _stable_seed_pool(
    images: list[dict[str, Any]],
) -> list[str]:
    seed_percentage = int(config["ranking"]["seed_percentage"])
    seed_size = max(1, len(images) * seed_percentage // 100)
    by_comps = sorted(
        images, key=lambda img: int(img["comparison_count"]), reverse=True
    )
    result = [img["filename"] for img in by_comps[:seed_size]]

    return result


def _pair_key(a: str, b: str) -> tuple[str, str]:
    _start = time.perf_counter()
    result: tuple[str, str] = (a, b) if a <= b else (b, a)
    return result


def _existing_pairs() -> set[tuple[str, str]]:
    result = {
        _pair_key(comp["filename_a"], comp["filename_b"])
        for comp in get_all_comparisons(weight=1.0)
    }
    return result


def _component_id(filename: str) -> int | None:
    _start = time.perf_counter()
    comp = crystal_graph.get_component(node_id=filename)
    result = comp.id if comp else None

    return result


def _score_gap(a: dict[str, Any], b: dict[str, Any]) -> float:
    _start = time.perf_counter()
    result = abs(float(a["score"]) - float(b["score"]))

    return result


def _pair_probability(a: dict[str, Any], b: dict[str, Any]) -> float:
    _start = time.perf_counter()
    result = expected_win_probability(rating_from_row(a), rating_from_row(b))

    return result


def _find_unseen_candidates(
    source: dict[str, Any],
    candidates: list[dict[str, Any]],
    pair_set: set[tuple[str, str]],
) -> Iterator[dict[str, Any]]:

    _start = time.perf_counter()
    source_name = source["filename"]
    results = 0
    for candidate in candidates:
        if (
            _pair_key(source_name, candidate["filename"]) not in pair_set
            and candidate["filename"] != source_name
        ):
            results += 1
            yield candidate

    logger.debug(f"find unseen candidates length:{results}", start_timer=_start)

    # result = [
    #     candidate
    #     for candidate in candidates
    #     if _score_gap(source, candidate) < score_gap
    #     and _prefer_cross_path(source["filename"], candidate["filename"]) == 1
    #     # and _pair_key(source_name, candidate["filename"]) not in pair_set
    #     and candidate["filename"] != source_name
    # ]

    # return result


def _are_in_different_paths(a: str, b: str) -> bool:
    _start = time.perf_counter()
    result = not crystal_graph.are_in_same_path(a, b)
    return result


def _build_low_count_pool(
    candidate_images: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    insertion_target = int(config["ranking"]["insertion_target_comparisons"])
    reserve_count = int(config["ranking"]["reserve_count"])

    for threshold in range(0, insertion_target + 1):
        pool = [
            img for img in candidate_images if int(img["comparison_count"]) <= threshold
        ]
        if len(pool) >= max(threshold + 2, reserve_count):
            # print("a", len(pool) , threshold,insertion_target,reserve_count)
            return pool
    result = [
        img
        for img in candidate_images
        if int(img["comparison_count"]) <= insertion_target
    ]
    return result


def _phase1_seed_coverage(
    seed_candidates: list[dict[str, Any]],
    existing_pair_set: set[tuple[str, str]],
) -> tuple[str, str] | None:
    _start = time.perf_counter()
    seed_target = int(config["ranking"]["seed_target_comparisons"])
    # p1_tiebreaker = _random_tiebreaker(seed_candidates)
    under_seed_target = sorted(
        [img for img in seed_candidates if int(img["comparison_count"]) < seed_target],
        key=lambda img: (
            int(img["comparison_count"]),
            -float(img["rating_sigma"]),
            # p1_tiebreaker[img["filename"]],
        ),
    )
    for source in under_seed_target:
        logger.debug(f"starting iterator", start_timer=_start)
        opponents: Iterator[dict[str, Any]] = _find_unseen_candidates(
            source, seed_candidates, existing_pair_set
        )
        # logger.debug("finish iterator", start_timer=_start)

        # opp_tiebreaker = _random_tiebreaker(opponents)
        chosen = None
        i = 0
        for opp in opponents:
            #  logger.debug(f"opponent {i}", start_timer=_start)
            i += 1
            if chosen is None:
                # if i < 3 and not _are_in_different_paths(
                #     source["filename"], opp["filename"]
                # ):
                #     continue
                chosen = opp

            if (
                i > 5
                and _score_gap(source, chosen) < 0.05
                and int(source["comparison_count"]) <= chosen["comparison_count"] + 1
            ):
                logger.debug(f"good candidate found st {i} steps", start_timer=_start)
                break

            if i > 10:
                break

            if int(opp["comparison_count"]) < chosen["comparison_count"]:
                chosen = opp
                continue
            if _score_gap(source, opp) < _score_gap(source, chosen):
                chosen = opp
                continue

        if chosen is None:
            continue
        result = (source["filename"], chosen["filename"])
        logger.debug(f"return result after {i} steps", start_timer=_start)
        return result
    return None


def _phase2_anchor_insert(
    candidate_images: list[dict[str, Any]],
    seed_pool: set[str],
    existing_pair_set: set[tuple[str, str]],
) -> tuple[str, str] | None:
    _start = time.perf_counter()
    pool = _build_low_count_pool(
        [img for img in candidate_images if img["filename"] not in seed_pool]
    )
    reserve_count = config["ranking"]["reserve_count"]
    if len(pool) < reserve_count:
        logger.warning(
            f"_phase2_anchor_insert: pool too small ({len(pool)} < {reserve_count})",
            start_timer=_start,
        )
        return None

    pool.sort(key=lambda img: (int(img["comparison_count"]), float(img["score"])))
    source = pool[0]
    source_name = source["filename"]
    source_mu = float(source["rating_mu"])

    remaining = [img for img in pool if img["filename"] != source_name]
    remaining.sort(key=lambda opp: (abs(float(opp["rating_mu"]) - source_mu),))
    opponents = _find_unseen_candidates(source, remaining, existing_pair_set)
    seen_opponents = 0
    for opponent in opponents:
        seen_opponents += 1
        opp_name = opponent["filename"]
        if not _are_in_different_paths(source_name, opp_name):
            continue
        result = (source_name, opp_name)
        # logger.debug(f"returning result: {result}")
        return result
    logger.debug(f"no pair found out of {seen_opponents} opponents")
    return None


def _phase3_collapsible_pairs(
    candidate_images: list[dict[str, Any]],
    pair_set: set[tuple[str, str]],
) -> tuple[str, str] | None:
    _start = time.perf_counter()
    only_wins = get_images_with_only_wins()
    only_loses = get_images_with_only_losses()
    if len(only_wins) == 1 and len(only_loses) == 1:
        return None

    candidate_names = {img["filename"] for img in candidate_images}

    chains_list = crystal_graph.get_all_chains()
    chains: list[ChainProxy] = [c[0] for c in chains_list]
    tops_by_comp: dict[int | None, dict[str, int]] = defaultdict(dict)
    bottoms_by_comp: dict[int | None, dict[str, int]] = defaultdict(dict)

    for chain in chains:
        if (
            chain.first
            and chain.first.filename in candidate_names
            and chain.first.is_top()
        ):

            comp = _component_id(chain.first.filename)
            if chain.first.filename not in tops_by_comp[comp]:
                tops_by_comp[comp][chain.first.filename] = len(chain.first.get_links())
        if (
            chain.last
            and chain.last.filename in candidate_names
            and chain.last.is_bottom()
        ):
            comp = _component_id(chain.last.filename)
            if chain.last.filename not in bottoms_by_comp[comp]:
                bottoms_by_comp[comp][chain.last.filename] = len(chain.last.get_links())
    for comp_id, bottoms in bottoms_by_comp.items():

        if len(bottoms) < 2:
            continue

        sorted_bottom_dict: list[tuple[str, int]] = list(bottoms.items())
        sorted_bottom_dict.sort(key=lambda bottom: bottom[1], reverse=False)
        sorted_bottoms: list[str] = [bottom[0] for bottom in sorted_bottom_dict]

        for i, a in enumerate(sorted_bottoms):
            if a not in only_loses:
                raise RuntimeError(f"bottom node {a} not present in only loses.")

            for b in sorted_bottoms[i + 1 :]:
                if b not in only_loses:
                    raise RuntimeError(f"bottom node {b} not present in only loses.")

                result = (a, b)

                return result

    for comp_id, tops in tops_by_comp.items():
        if len(tops.items()) < 2:
            continue
        logger.debug(f"id:{comp_id}, total extremes:{len(tops.items())}")

        sorted_top_dict: list[tuple[str, int]] = list(tops.items())
        sorted_top_dict.sort(key=lambda top: top[1], reverse=False)
        sorted_tops: list[str] = [top[0] for top in sorted_top_dict]

        for i, a in enumerate(sorted_tops):
            if a not in only_wins:
                raise RuntimeError(f"top node {a} not present in only wins.")
            for b in sorted_tops[i + 1 :]:
                if b not in only_wins:
                    raise RuntimeError(f"top node {b} not present in only wins.")

                result = (a, b)

                return result

    return None


_last_chains_index: list[int] = []


def _phase4_chain_merge(
    candidate_images: list[dict[str, Any]],
) -> tuple[str, str] | None:
    global _last_chains_index
    score_threshold = 0.01
    min_comparisons = int(config["ranking"]["insertion_target_comparisons"])

    if len(_last_chains_index) > MIN_CHAIN_THRESHOLD:
        # logger.debug(f"last chains before:{_last_chains_index}")
        _last_chains_index = _last_chains_index[MIN_CHAIN_THRESHOLD // 2 :]
        # logger.debug(f"last chains after:{_last_chains_index}")

    _start = time.perf_counter()
    candidate_names = {img["filename"] for img in candidate_images}
    # logger.warning(f"candidate_names", start_timer=_start)

    chains_list: list[tuple[ChainProxy, list[NodeTuple]]] = (
        crystal_graph.get_all_chains(min_length=1, sort_order="asc")
    )
    chains: list[list[NodeTuple]] = [c[1] for c in chains_list]

    if len(chains) < MIN_CHAIN_THRESHOLD:
        logger.info(
            f"skipping phase 4: <{MIN_CHAIN_THRESHOLD} chains", start_timer=_start
        )
        return None

    logger.debug(f"shortest chain: {len(chains[0])}, longest: {len(chains[-1])}")
    top_n: int = min(MIN_CHAIN_THRESHOLD * 10, len(chains))

    for i in range(top_n - 1):
        if i in _last_chains_index:
            continue
        a_nodes: list[NodeProxy] = [
            n[0]
            for n in chains[i]
            if n[0].filename in candidate_names
            and n[1]
            and n[0].comparison_count > min_comparisons
        ]
        if len(a_nodes) == 0:
            continue
        a_mid: NodeProxy = a_nodes[len(a_nodes) // 2]

        for j in range(len(chains) - 1, i, -1):
            if j in _last_chains_index:
                continue
            if i == j:
                continue
            b_nodes: list[NodeProxy] = [
                n[0] for n in chains[j] if n[0].filename in candidate_names and n[1]
            ]

            if len(b_nodes) == 0:
                continue

            b_mid: NodeProxy = b_nodes[len(b_nodes) // 2]
            pair_list: list[tuple[NodeProxy, NodeProxy]] = []
            pair_list.insert(0, (a_mid, b_mid))
            pair_list.extend(list(zip(a_nodes, b_nodes)))
            pair_list.extend(
                [(a, b) for a in a_nodes for b in b_nodes if a.filename != b.filename]
            )
            for a, b in set(pair_list):
                a_name = a.filename
                b_name = b.filename
                if abs(a.score - b.score) > score_threshold:
                    continue

                if crystal_graph.are_in_same_path(a_name, b_name):
                    continue

                result = (a_name, b_name)
                _last_chains_index.append(i)
                _last_chains_index.append(j)
                logger.debug(f"I={i},j={j}", start_timer=_start)
                logger.debug(
                    f"chain i={len(a_nodes)}({len(chains[i])}),chain j={len(b_nodes)}({len(chains[j])})",
                    start_timer=_start,
                )

                return result
    logger.warning(
        f"skipping phase 4: no valid pair found in shorter {MIN_CHAIN_THRESHOLD*10} chains",
        start_timer=_start,
    )

    return None


def _phase5_uncertainty_refine(
    candidate_images: list[dict[str, Any]],
    pair_set: set[tuple[str, str]],
) -> tuple[str, str] | None:
    _start = time.perf_counter()
    # Sort by highest sigma (uncertainty) first
    uncertainty_pool = sorted(
        candidate_images,
        key=lambda img: (-float(img["rating_sigma"]), int(img["comparison_count"])),
    )

    # Build seed pool for similar-mu matching
    seed_filenames = set(_stable_seed_pool(candidate_images))
    seed_pool = [img for img in candidate_images if img["filename"] in seed_filenames]
    if not seed_pool:
        return None

    for source in uncertainty_pool:
        source_mu = float(source["rating_mu"])
        source_name = source["filename"]

        # Find seed images with similar mu, unseen with source
        candidates = [
            s for s in seed_pool
            if s["filename"] != source_name
            and _pair_key(source_name, s["filename"]) not in pair_set
            and not crystal_graph.are_in_same_path(source_name, s["filename"])
        ]

        if not candidates:
            continue

        chosen = min(
            candidates,
            key=lambda s: abs(float(s["rating_mu"]) - source_mu),
        )

        result = (source_name, chosen["filename"])
        return result

    return None


def _phase_fallback(
    candidate_images: list[dict[str, Any]],
    pair_set: set[tuple[str, str]],
) -> tuple[str, str] | None:
    _start = time.perf_counter()
    ordered = sorted(
        candidate_images,
        key=lambda img: (int(img["comparison_count"]),),
    )
    for idx, left in enumerate(ordered):
        for right in ordered[idx + 1 :]:
            if _pair_key(left["filename"], right["filename"]) in pair_set:
                continue

            result = (left["filename"], right["filename"])

            return result
    return None



