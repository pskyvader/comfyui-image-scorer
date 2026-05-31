"""Active pair selection for the TrueSkill-based step01 flow."""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
import time
from typing import Any
import logging
import random

from shared.graph import crystal_graph
from shared.config import config
from shared.logger import SharedLogger
from external_modules.database_structure.comparisons_table import (
    get_all_comparisons,
    get_total_comparisons,
    comparison_exists_for_pair,
)

from .trueskill_rating import Rating, expected_win_probability, rating_from_row

logger = SharedLogger.get_logger(__name__)

_skip_before: int = 0


@contextmanager
def _log_timing(label: str):
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    # logger.info("TIMING [%s]: %.3f s", label, elapsed)


def _stable_seed_pool(
    images: list[dict[str, Any]],
) -> list[str]:
    seed_size = int(config["ranking"]["seed_size"])
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


def _prefer_cross_path(a: str, b: str) -> int:
    _start = time.perf_counter()
    result = 0 if crystal_graph.are_in_same_path(a, b) else 1

    return result


def _random_tiebreaker(images: list[dict[str, Any]]) -> dict[str, float]:
    _start = time.perf_counter()
    result = {img["filename"]: random.random() for img in images}

    return result


def _find_unseen_candidates(
    source: dict[str, Any],
    candidates: list[dict[str, Any]],
    pair_set: set[tuple[str, str]],
) -> list[dict[str, Any]]:
    source_name = source["filename"]
    result = [
        candidate
        for candidate in candidates
        if candidate["filename"] != source_name
        and _pair_key(source_name, candidate["filename"]) not in pair_set
    ]
    return result


def _are_in_different_paths(a: str, b: str) -> bool:
    _start = time.perf_counter()
    result = not crystal_graph.are_in_same_path(a, b)
    return result


def _build_low_count_pool(
    candidate_images: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    _start = time.perf_counter()
    seed_target = int(config["ranking"]["seed_target_comparisons"])
    for threshold in range(0, seed_target + 1, 2):
        pool = [
            img for img in candidate_images if int(img["comparison_count"]) <= threshold
        ]
        if len(pool) >= 2:
            return pool
    result = [
        img for img in candidate_images if int(img["comparison_count"]) <= seed_target
    ]
    return result


def _phase1_seed_coverage(
    seed_candidates: list[dict[str, Any]],
    existing_pair_set: set[tuple[str, str]],
) -> tuple[tuple[str, str] | None, dict[str, Any]]:
    _start = time.perf_counter()
    seed_target = int(config["ranking"]["seed_target_comparisons"])
    p1_tiebreaker = _random_tiebreaker(seed_candidates)
    under_seed_target = sorted(
        [img for img in seed_candidates if int(img["comparison_count"]) < seed_target],
        key=lambda img: (
            int(img["comparison_count"]),
            -float(img["rating_sigma"]),
            p1_tiebreaker[img["filename"]],
        ),
    )
    for source in under_seed_target:
        opponents = _find_unseen_candidates(source, seed_candidates, existing_pair_set)
        if not opponents:
            continue
        opp_tiebreaker = _random_tiebreaker(opponents)
        opponents.sort(
            key=lambda opp: (
                int(opp["comparison_count"]),
                -_prefer_cross_path(source["filename"], opp["filename"]),
                _score_gap(source, opp),
                opp_tiebreaker[opp["filename"]],
            )
        )
        chosen = opponents[0]

        result = (
            (source["filename"], chosen["filename"]),
            {
                "pair_type": "bootstrap_seed",
                "left_comp_count": int(source["comparison_count"]),
                "right_comp_count": int(chosen["comparison_count"]),
                "refinement_details": None,
            },
        )
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
    if len(pool) < 2:
        logger.warning(
            f"_phase2_anchor_insert: pool too small ({time.perf_counter() - _start:.4f}s)"
        )
        return None, {}

    pool.sort(key=lambda img: int(img["comparison_count"]))
    source = pool[0]
    source_name = source["filename"]
    source_mu = float(source["rating_mu"])
    source_comp = _component_id(source_name)

    remaining = [img for img in pool if img["filename"] != source_name]
    remaining.sort(
        key=lambda opp: (
            0 if _component_id(opp["filename"]) != source_comp else 1,
            abs(float(opp["rating_mu"]) - source_mu),
        )
    )
    opponents = _find_unseen_candidates(source, remaining, existing_pair_set)

    for opponent in opponents:
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
        return result

    return None, {}


def _phase3_collapsible_pairs(
    candidate_images: list[dict[str, Any]],
    pair_set: set[tuple[str, str]],
) -> tuple[tuple[str, str] | None, dict[str, Any]]:
    _start = time.perf_counter()
    candidate_names = {img["filename"] for img in candidate_images}

    chains = crystal_graph.get_all_chains()
    tops_by_comp: dict[int | None, list[str]] = defaultdict(list)
    bottoms_by_comp: dict[int | None, list[str]] = defaultdict(list)

    for chain in chains:
        if chain.first and chain.first.filename in candidate_names:
            comp = _component_id(chain.first.filename)
            tops_by_comp[comp].append(chain.first.filename)
        if chain.last and chain.last.filename in candidate_names:
            comp = _component_id(chain.last.filename)
            bottoms_by_comp[comp].append(chain.last.filename)

    for comp_id, tops in tops_by_comp.items():
        if len(tops) < 2:
            continue
        for i, a in enumerate(tops):
            for b in tops[i + 1 :]:
                if _pair_key(a, b) in pair_set:
                    continue
                if crystal_graph.are_in_same_path(a, b):
                    continue

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
        for i, a in enumerate(bottoms):
            for b in bottoms[i + 1 :]:
                if _pair_key(a, b) in pair_set:
                    continue
                if crystal_graph.are_in_same_path(a, b):
                    continue

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


def _phase4_chain_merge(
    candidate_images: list[dict[str, Any]],
    pair_set: set[tuple[str, str]],
) -> tuple[tuple[str, str] | None, dict[str, Any]]:
    _start = time.perf_counter()
    candidate_names = {img["filename"] for img in candidate_images}

    chains = [
        c
        for c in crystal_graph.get_all_chains()
        if c.length >= 3 and any(n.filename in candidate_names for n in c.nodes)
    ]
    if len(chains) < 2:
        logger.warning(
            f"_phase4_chain_merge: <2 chains ({time.perf_counter() - _start:.4f}s)"
        )
        return None, {}

    chains.sort(key=lambda c: c.length, reverse=True)
    top_n = min(20, len(chains))

    for i in range(top_n):
        a_nodes = [n for n in chains[i].nodes if n.filename in candidate_names]
        if len(a_nodes) < 2:
            continue
        a = a_nodes[len(a_nodes) // 2]
        a_name = a.filename

        for j in range(len(chains)):
            if i == j:
                continue
            b_nodes = [n for n in chains[j].nodes if n.filename in candidate_names]
            if len(b_nodes) < 2:
                continue
            b = b_nodes[len(b_nodes) // 2]
            b_name = b.filename

            if _pair_key(a_name, b_name) in pair_set:
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
                        "source_chain_length": chains[i].length,
                        "opponent_chain_length": chains[j].length,
                    },
                },
            )

            return result

    return None, {}


def _phase5_uncertainty_refine(
    candidate_images: list[dict[str, Any]],
    pair_set: set[tuple[str, str]],
) -> tuple[tuple[str, str] | None, dict[str, Any]]:
    _start = time.perf_counter()
    p3_tiebreaker = _random_tiebreaker(candidate_images)
    uncertainty_pool = sorted(
        candidate_images,
        key=lambda img: (
            -float(img["rating_sigma"]),
            int(img["comparison_count"]),
            p3_tiebreaker[img["filename"]],
        ),
    )

    for source in uncertainty_pool:
        unseen = _find_unseen_candidates(source, candidate_images, pair_set)
        if not unseen:
            continue
        source_comp = _component_id(source["filename"])
        source_rating = rating_from_row(source)
        unseen_tiebreaker = _random_tiebreaker(unseen)
        unseen.sort(
            key=lambda opp: (
                abs(
                    expected_win_probability(source_rating, rating_from_row(opp)) - 0.5
                ),
                0 if _component_id(opp["filename"]) != source_comp else 1,
                0 if _prefer_cross_path(source["filename"], opp["filename"]) else 1,
                abs(float(opp["rating_sigma"]) - float(source["rating_sigma"])),
                unseen_tiebreaker[opp["filename"]],
            )
        )
        chosen = unseen[0]
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
    fb_tiebreaker = _random_tiebreaker(candidate_images)
    ordered = sorted(
        candidate_images,
        key=lambda img: (
            int(img["comparison_count"]),
            fb_tiebreaker[img["filename"]],
        ),
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
    total_comps = get_total_comparisons()
    if total_comps % reserve_count == 0:
        _skip_before = 0

    if _skip_before <= 0:
        with _log_timing("phase1_seed_coverage"):
            result = _phase1_seed_coverage(seed_candidates, existing_pairs_set)
        if result[0]:
            _skip_before = 0
            return result

    if _skip_before <= 1:
        with _log_timing("phase2_anchor_insert"):
            result = _phase2_anchor_insert(
                candidate_images, seed_pool, existing_pairs_set
            )
        if result[0]:
            _skip_before = 1
            return result

    if _skip_before <= 2:
        with _log_timing("phase3_collapsible_pairs"):
            result = _phase3_collapsible_pairs(candidate_images, existing_pairs_set)
        if result[0]:
            _skip_before = 2
            return result

    if _skip_before <= 3:
        with _log_timing("phase4_chain_merge"):
            result = _phase4_chain_merge(candidate_images, existing_pairs_set)
        if result[0]:
            _skip_before = 3
            return result

    if _skip_before <= 4:
        with _log_timing("phase5_uncertainty_refine"):
            result = _phase5_uncertainty_refine(candidate_images, existing_pairs_set)
        if result[0]:
            _skip_before = 4
            return result

    if _skip_before <= 5:
        with _log_timing("phase_fallback"):
            result = _phase_fallback(candidate_images, existing_pairs_set)
        if result[0]:
            _skip_before = 5
            return result

    logger.warning("select_pair: no pair found after all phases")
    return None, {}
