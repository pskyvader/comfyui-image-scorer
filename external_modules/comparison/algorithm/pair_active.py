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
    get_total_comparisons,
    get_images_with_only_wins,
    get_images_with_only_losses,
)

from .trueskill_rating import expected_win_probability, rating_from_row

logger = SharedLogger.get_logger(__name__)

_skip_before: int = 0


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
) -> tuple[tuple[str, str] | None, dict[str, Any]]:
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
        result = (
            (source["filename"], chosen["filename"]),
            {
                "pair_type": "bootstrap_seed",
                "left_comp_count": int(source["comparison_count"]),
                "right_comp_count": int(chosen["comparison_count"]),
                "refinement_details": None,
            },
        )
        logger.debug(f"return result after {i} steps", start_timer=_start)
        return result
    return None, {}


def _phase2_anchor_insert(
    candidate_images: list[dict[str, Any]],
    seed_pool: set[str],
    existing_pair_set: set[tuple[str, str]],
) -> tuple[tuple[str, str] | None, dict[str, Any]]:
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
        return None, {}

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
        result = (
            (source_name, opp_name),
            {
                "pair_type": "anchor_insert",
                "left_comp_count": int(source["comparison_count"]),
                "right_comp_count": int(opponent["comparison_count"]),
                "refinement_details": {
                    "source_mu": round(source_mu, 4),
                    "opponent_mu": round(float(opponent["rating_mu"]), 4),
                },
            },
        )
        # logger.debug(f"returning result: {result}")
        return result
    logger.debug(f"no pair found out of {seen_opponents} opponents")
    return None, {}


def _phase3_collapsible_pairs(
    candidate_images: list[dict[str, Any]],
    pair_set: set[tuple[str, str]],
) -> tuple[tuple[str, str] | None, dict[str, Any]]:
    _start = time.perf_counter()
    only_wins = get_images_with_only_wins()
    only_loses = get_images_with_only_losses()
    if len(only_wins) == 1 and len(only_loses) == 1:
        return None, {}

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

                result = (
                    (a, b),
                    {
                        "pair_type": "collapsible",
                        "left_comp_count": 0,
                        "right_comp_count": 0,
                        "refinement_details": {
                            "collapsible_type": "both_top",
                            "component": comp_id,
                        },
                    },
                )

                return result

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

                result = (
                    (a, b),
                    {
                        "pair_type": "collapsible",
                        "left_comp_count": 0,
                        "right_comp_count": 0,
                        "refinement_details": {
                            "collapsible_type": "both_bottom",
                            "component": comp_id,
                        },
                    },
                )

                return result

    return None, {}


_last_chains_index: list[int] = []


def _phase4_chain_merge(
    candidate_images: list[dict[str, Any]],
) -> tuple[tuple[str, str] | None, dict[str, Any]]:
    global _last_chains_index
    min_chain_threshold = 20
    score_threshold = 0.01
    min_comparisons = int(config["ranking"]["insertion_target_comparisons"])

    if len(_last_chains_index) > min_chain_threshold:
        # logger.debug(f"last chains before:{_last_chains_index}")
        _last_chains_index = _last_chains_index[min_chain_threshold // 2 :]
        # logger.debug(f"last chains after:{_last_chains_index}")

    _start = time.perf_counter()
    candidate_names = {img["filename"] for img in candidate_images}
    # logger.warning(f"candidate_names", start_timer=_start)

    chains_list: list[tuple[ChainProxy, list[NodeTuple]]] = (
        crystal_graph.get_all_chains(min_length=3, sort_order="asc")
    )
    chains: list[list[NodeTuple]] = [c[1] for c in chains_list]

    if len(chains) < min_chain_threshold:
        logger.info(
            f"skipping phase 4: <{min_chain_threshold} chains", start_timer=_start
        )
        return None, {}

    logger.debug(f"shortest chain: {len(chains[0])}, longest: {len(chains[-1])}")
    top_n: int = min(min_chain_threshold * 10, len(chains))

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

                result = (
                    (a_name, b_name),
                    {
                        "pair_type": "chain_merge",
                        "left_comp_count": 0,
                        "right_comp_count": 0,
                        "refinement_details": {
                            "source_chain_length": len(chains[i]),
                            "opponent_chain_length": len(chains[j]),
                        },
                    },
                )
                _last_chains_index.append(i)
                _last_chains_index.append(j)
                logger.debug(f"I={i},j={j}", start_timer=_start)
                logger.debug(
                    f"chain i={len(a_nodes)}({len(chains[i])}),chain j={len(b_nodes)}({len(chains[j])})",
                    start_timer=_start,
                )

                return result
    logger.warning(
        f"skipping phase 4: no valid pair found in shorter {min_chain_threshold*10} chains",
        start_timer=_start,
    )

    return None, {}


def _phase5_uncertainty_refine(
    candidate_images: list[dict[str, Any]],
    pair_set: set[tuple[str, str]],
) -> tuple[tuple[str, str] | None, dict[str, Any]]:
    _start = time.perf_counter()
    uncertainty_pool = sorted(
        candidate_images,
        key=lambda img: (-float(img["rating_sigma"]), int(img["comparison_count"])),
    )

    for source in uncertainty_pool:
        source_rating = rating_from_row(source)

        unseen_iterator = _find_unseen_candidates(source, candidate_images, pair_set)

        chosen = None
        i = 0
        for opp in unseen_iterator:
            i += 1
            a_name = source["filename"]
            b_name = opp["filename"]
            if crystal_graph.are_in_same_path(a_name, b_name):
                continue
            if chosen is None:
                chosen = opp
            else:
                if abs(source["rating_mu"] - opp["rating_mu"]) < abs(
                    source["rating_mu"] - chosen["rating_mu"]
                ):
                    chosen = opp
            if i >= 5:
                break

        if chosen is None:
            continue

        prob = _pair_probability(source, chosen)

        result = (
            (source["filename"], chosen["filename"]),
            {
                "pair_type": "uncertainty_refine",
                "left_comp_count": int(source["comparison_count"]),
                "right_comp_count": int(chosen["comparison_count"]),
                "refinement_details": {
                    "strategy": "uncertainty",
                    "predicted_win_probability": round(prob, 4),
                    "left_sigma": round(float(source["rating_sigma"]), 4),
                    "right_sigma": round(float(chosen["rating_sigma"]), 4),
                },
            },
        )

        return result

    return None, {}


def _phase_fallback(
    candidate_images: list[dict[str, Any]],
    pair_set: set[tuple[str, str]],
) -> tuple[tuple[str, str] | None, dict[str, Any]]:
    _start = time.perf_counter()
    ordered = sorted(
        candidate_images,
        key=lambda img: (int(img["comparison_count"]),),
    )
    for idx, left in enumerate(ordered):
        for right in ordered[idx + 1 :]:
            if _pair_key(left["filename"], right["filename"]) in pair_set:
                continue

            result = (
                (left["filename"], right["filename"]),
                {
                    "pair_type": "fallback",
                    "left_comp_count": int(left["comparison_count"]),
                    "right_comp_count": int(right["comparison_count"]),
                    "refinement_details": None,
                },
            )

            return result
    return None, {}


def reset_skip():
    global _skip_before
    _skip_before = 0


def select_pair(
    all_images: list[dict[str, Any]],
    candidate_images: list[dict[str, Any]],
) -> tuple[tuple[str, str] | None, dict[str, Any]]:
    global _skip_before

    if len(candidate_images) < 2:
        logger.warning(
            "select_pair: only %d candidates, need >=2", len(candidate_images)
        )
        return None, {}

    existing_pairs_set = _existing_pairs()

    seed_pool = set(_stable_seed_pool(all_images))

    # randomize order of candidates

    random.shuffle(candidate_images)

    seed_candidates = [img for img in candidate_images if img["filename"] in seed_pool]

    reserve_count = int(config["ranking"]["reserve_count"])
    total_comps: int = get_total_comparisons()
    logger.debug(f"total comps: {total_comps}, skip before:{_skip_before}")
    if total_comps % reserve_count == 0:
        reset_skip()

    if _skip_before <= 0:
        result = _phase1_seed_coverage(seed_candidates, existing_pairs_set)
        if result[0]:
            _skip_before = 0
            return result

    if _skip_before <= 1:
        result = _phase2_anchor_insert(candidate_images, seed_pool, existing_pairs_set)
        if result[0]:
            _skip_before = 1
            return result

    if _skip_before <= 2:
        result = _phase3_collapsible_pairs(candidate_images, existing_pairs_set)
        if result[0]:
            _skip_before = 2
            return result

    if _skip_before <= 3:
        result = _phase4_chain_merge(candidate_images)
        if result[0]:
            _skip_before = 3
            return result

    if _skip_before <= 4:
        result = _phase5_uncertainty_refine(candidate_images, existing_pairs_set)
        if result[0]:
            _skip_before = 4
            return result

    if _skip_before <= 5:
        result = _phase_fallback(candidate_images, existing_pairs_set)
        if result[0]:
            _skip_before = 5
            return result

    logger.warning("select_pair: no pair found after all phases")
    return None, {}
