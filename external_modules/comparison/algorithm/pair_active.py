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
from external_modules.database_structure.comparisons_table import get_all_comparisons, get_total_comparisons
from .trueskill_rating import Rating, expected_win_probability, rating_from_row

logger: logging.Logger = logging.getLogger(__name__)

_skip_before: int = 0  # phases before this index are skipped on non-reserve calls


@contextmanager
def _log_timing(str):
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info("TIMING [%s]: %.3f s", label, elapsed)


def _stable_seed_pool(
    _start = time.perf_counter()
    _start = time.perf_counter()
    images: list[dict[str, Any]] | list[str],
    logger.debug("_stable_seed_pool took %.4fs", time.perf_counter() - _start)
) -> list[str]:
    seed_size = int(config["ranking"]["seed_size"])
    if images and isinstance(images[0], str):
        return images[:seed_size]
    by_comps = sorted(
        images, key=lambda img: int(img["comparison_count"]), reverse=True
    )
    return [img["filename"] for img in by_comps[:seed_size]]


def _pair_key(str, b: str) -> tuple[str, str]:
    result = tuple(sorted((a, b)))
    logger.debug("_pair_key took %.4fs", time.perf_counter() - _start)
    return result


def _existing_pairs({:
        _pair_key(comp["filename_a"], comp["filename_b"])
        for comp in get_all_comparisons(weight=1.0)
    }


def _component_id(str) -> int | None:
    comp = crystal_graph.get_component(node_id=filename)
    result = comp.id if comp else None
    logger.debug("_component_id took %.4fs", time.perf_counter() - _start)
    return result


def _score_gap(dict[str, Any], b: dict[str, Any]) -> float:
    result = abs(float(a["score"]) - float(b["score"]))
    logger.debug("_score_gap took %.4fs", time.perf_counter() - _start)
    return result


def _pair_probability(dict[str, Any], b: dict[str, Any]) -> float:
    result = expected_win_probability(rating_from_row(a), rating_from_row(b))
    logger.debug("_pair_probability took %.4fs", time.perf_counter() - _start)
    return result


def _prefer_cross_path(str, b: str) -> int:
    try:
        result = 0 if crystal_graph.are_in_same_path(a, b) else 1
        logger.debug("_prefer_cross_path took %.4fs", time.perf_counter() - _start)
        return result
    except Exception:
        result = 1
        logger.debug("_prefer_cross_path took %.4fs", time.perf_counter() - _start)
        return result


def _random_tiebreaker(list[dict[str, Any]]) -> dict[str, float]:
    result = {img["filename"]: random.random() for img in images}
    logger.debug("_random_tiebreaker took %.4fs", time.perf_counter() - _start)
    return result


def _find_unseen_candidates(
    _start = time.perf_counter()
    _start = time.perf_counter()
    source: dict[str, Any],
    candidates: list[dict[str, Any]],
    pair_set: set[tuple[str, str]],
    logger.debug("_find_unseen_candidates took %.4fs", time.perf_counter() - _start)
) -> list[dict[str, Any]]:
    source_name = source["filename"]
    return [
        candidate
        for candidate in candidates
        if candidate["filename"] != source_name
        and _pair_key(source_name, candidate["filename"]) not in pair_set
    ]


def _are_in_different_paths(str, b: str) -> bool:
    try:
        result = not crystal_graph.are_in_same_path(a, b)
        logger.debug("_are_in_different_paths took %.4fs", time.perf_counter() - _start)
        return result
    except Exception:
        result = True
        logger.debug("_are_in_different_paths took %.4fs", time.perf_counter() - _start)
        return result


def _build_low_count_pool(
    _start = time.perf_counter()
    _start = time.perf_counter()
    candidate_images: list[dict[str, Any]],
    logger.debug("_build_low_count_pool took %.4fs", time.perf_counter() - _start)
) -> list[dict[str, Any]]:
    seed_target = int(config["ranking"]["seed_target_comparisons"])
    for threshold in range(0, seed_target + 1, 2):
        pool = [
            img
            for img in candidate_images
            if int(img["comparison_count"]) <= threshold
        ]
        if len(pool) >= 2:
            return pool
    return [
        img
        for img in candidate_images
        if int(img["comparison_count"]) <= seed_target
    ]


def _phase1_seed_coverage(
    _start = time.perf_counter()
    _start = time.perf_counter()
    seed_candidates: list[dict[str, Any]],
    pair_set: set[tuple[str, str]],
    logger.debug("_phase1_seed_coverage took %.4fs", time.perf_counter() - _start)
) -> tuple[tuple[str, str] | None, dict[str, Any]]:
    seed_target = int(config["ranking"]["seed_target_comparisons"])
    p1_tiebreaker = _random_tiebreaker(seed_candidates)
    under_seed_target = sorted(
        [
            img
            for img in seed_candidates
            if int(img["comparison_count"]) < seed_target
        ],
        key=lambda img: (
            int(img["comparison_count"]),
            -float(img["rating_sigma"]),
            p1_tiebreaker[img["filename"]],
        ),
    )
    for source in under_seed_target:
        opponents = _find_unseen_candidates(source, seed_candidates, pair_set)
        if not opponents:
            logger.debug(
                "Phase1: no unseen opponents for %s among seeds", source["filename"]
            )
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
        logger.info(
            "Phase1: selected %s vs %s (source comps=%d, sigma=%.3f, chosen comps=%d, score_gap=%.4f)",
            source["filename"],
            chosen["filename"],
            int(source["comparison_count"]),
            float(source["rating_sigma"]),
            int(chosen["comparison_count"]),
            _score_gap(source, chosen),
        )
        return (
            (source["filename"], chosen["filename"]),
            {
                "pair_type": "bootstrap_seed",
                "left_comp_count": int(source["comparison_count"]),
                "right_comp_count": int(chosen["comparison_count"]),
                "refinement_details": None,
            },
        )
    logger.debug("Phase1: no pair found, moving to Phase2")
    return None, {}


def _phase2_anchor_insert(
    _start = time.perf_counter()
    _start = time.perf_counter()
    candidate_images: list[dict[str, Any]],
    seed_pool: set[str],
    logger.debug("_phase2_anchor_insert took %.4fs", time.perf_counter() - _start)
) -> tuple[tuple[str, str] | None, dict[str, Any]]:
    pool = _build_low_count_pool(
        [img for img in candidate_images if img["filename"] not in seed_pool]
    )
    if len(pool) < 2:
        logger.debug("Phase2: low-count pool too small, moving to Phase3")
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

    for opponent in remaining:
        opp_name = opponent["filename"]
        if not _are_in_different_paths(source_name, opp_name):
            continue
        logger.info(
            "Phase2: selected %s vs %s (source comps=%d, opp comps=%d, mu_diff=%.4f)",
            source_name,
            opp_name,
            int(source["comparison_count"]),
            int(opponent["comparison_count"]),
            abs(float(opponent["rating_mu"]) - source_mu),
        )
        return (
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

    logger.debug("Phase2: no pair found, moving to Phase3")
    return None, {}


def _phase3_collapsible_pairs(
    _start = time.perf_counter()
    _start = time.perf_counter()
    candidate_images: list[dict[str, Any]],
    pair_set: set[tuple[str, str]],
    logger.debug("_phase3_collapsible_pairs took %.4fs", time.perf_counter() - _start)
) -> tuple[tuple[str, str] | None, dict[str, Any]]:
    """Phase 3: find a collapsible pair — two tops or two bottoms in the same
    component, not transitively connected, not already compared.
    One click resolves the ranking order for both branches."""
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
                logger.info(
                    "Phase3 (collapsible): selected %s vs %s (both tops, comp=%s)",
                    a,
                    b,
                    comp_id,
                )
                return (
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

    for comp_id, bottoms in bottoms_by_comp.items():
        if len(bottoms) < 2:
            continue
        for i, a in enumerate(bottoms):
            for b in bottoms[i + 1 :]:
                if _pair_key(a, b) in pair_set:
                    continue
                if crystal_graph.are_in_same_path(a, b):
                    continue
                logger.info(
                    "Phase3 (collapsible): selected %s vs %s (both bottoms, comp=%s)",
                    a,
                    b,
                    comp_id,
                )
                return (
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

    logger.debug("Phase3: no collapsible pair found, moving to Phase4")
    return None, {}


def _phase4_chain_merge(
    _start = time.perf_counter()
    _start = time.perf_counter()
    candidate_images: list[dict[str, Any]],
    pair_set: set[tuple[str, str]],
    logger.debug("_phase4_chain_merge took %.4fs", time.perf_counter() - _start)
) -> tuple[tuple[str, str] | None, dict[str, Any]]:
    """Phase 4: reduce chain count by comparing internal nodes from the
    longest chains. Picks a mid-chain node (not the shared global top/bottom)
    from two different chains that are not transitively connected."""
logger = logging.getLogger(__name__)
    candidate_names = {img["filename"] for img in candidate_images}

    chains = [
        c
        for c in crystal_graph.get_all_chains()
        if c.length >= 3 and any(n.filename in candidate_names for n in c.nodes)
    ]
    if len(chains) < 2:
        logger.debug("Phase4: fewer than 2 chains with candidates, skipping")
        return None, {}

    chains.sort(key=lambda c: c.length, reverse=True)
    top_n = min(20, len(chains))

    for i in range(top_n):
        a_nodes = [n for n in chains[i].nodes if n.filename in candidate_names]
        if len(a_nodes) < 2:
            continue
        # Pick a node from the middle third to avoid global endpoints
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

            logger.info(
                "Phase4 (chain_merge): selected %s vs %s (chain_len=%d vs %d)",
                a_name,
                b_name,
                chains[i].length,
                chains[j].length,
            )
            return (
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

    logger.debug("Phase4: no chain merge pair found, moving to Phase5")
    return None, {}


def _phase5_uncertainty_refine(
    _start = time.perf_counter()
    _start = time.perf_counter()
    candidate_images: list[dict[str, Any]],
    pair_set: set[tuple[str, str]],
    logger.debug("_phase5_uncertainty_refine took %.4fs", time.perf_counter() - _start)
) -> tuple[tuple[str, str] | None, dict[str, Any]]:
    p3_tiebreaker = _random_tiebreaker(candidate_images)
    uncertainty_pool = sorted(
        candidate_images,
        key=lambda img: (
            -float(img["rating_sigma"]),
            int(img["comparison_count"]),
            p3_tiebreaker[img["filename"]],
        ),
    )
    logger.info(
        "Phase3 (uncertainty_refine): %d candidates, top sigma=%.4f (image=%s)",
        len(uncertainty_pool),
        float(uncertainty_pool[0]["rating_sigma"]) if uncertainty_pool else 0,
        uncertainty_pool[0]["filename"] if uncertainty_pool else "N/A",
    )
    for source in uncertainty_pool:
        unseen = _find_unseen_candidates(source, candidate_images, pair_set)
        if not unseen:
            logger.debug("Phase3: no unseen opponents for %s", source["filename"])
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
                abs(
                    float(opp["rating_sigma"])
                    - float(source["rating_sigma"])
                ),
                unseen_tiebreaker[opp["filename"]],
            )
        )
        chosen = unseen[0]
        prob = _pair_probability(source, chosen)
        logger.info(
            "Phase3: selected %s vs %s (sigma_src=%.4f, sigma_opp=%.4f, win_prob=%.4f, same_comp=%s, cross_path=%s)",
            source["filename"],
            chosen["filename"],
            float(source["rating_sigma"]),
            float(chosen["rating_sigma"]),
            prob,
            _component_id(chosen["filename"]) == source_comp,
            not _prefer_cross_path(source["filename"], chosen["filename"]),
        )
        return (
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
    logger.debug("Phase3: no pair found, moving to fallback")
    return None, {}


def _phase_fallback(
    _start = time.perf_counter()
    _start = time.perf_counter()
    candidate_images: list[dict[str, Any]],
    pair_set: set[tuple[str, str]],
    logger.debug("_phase_fallback took %.4fs", time.perf_counter() - _start)
) -> tuple[tuple[str, str] | None, dict[str, Any]]:
    fb_tiebreaker = _random_tiebreaker(candidate_images)
    ordered = sorted(
        candidate_images,
        key=lambda img: (
            int(img["comparison_count"]),
            fb_tiebreaker[img["filename"]],
        ),
    )
    logger.info("Fallback: scanning %d candidates for any unseen pair", len(ordered))
    for idx, left in enumerate(ordered):
        for right in ordered[idx + 1 :]:
            if _pair_key(left["filename"], right["filename"]) in pair_set:
                continue
            logger.info(
                "Fallback: selected %s vs %s (comps=%d, %d)",
                left["filename"],
                right["filename"],
                int(left["comparison_count"]),
                int(right["comparison_count"]),
            )
            return (
                (left["filename"], right["filename"]),
            {
                "pair_type": "fallback",
                "left_comp_count": int(left["comparison_count"]),
                "right_comp_count": int(right["comparison_count"]),
                "refinement_details": None,
            },
            )
    return None, {}


def select_pair(
    _start = time.perf_counter()
    _start = time.perf_counter()
    all_images: list[dict[str, Any]],
    candidate_images: list[dict[str, Any]],
    exclude_set: set[str] | None = None,
    logger.debug("select_pair took %.4fs", time.perf_counter() - _start)
) -> tuple[tuple[str, str] | None, dict[str, Any]]:
    global _skip_before

    if len(candidate_images) < 2:
        logger.warning(
            "select_pair: only %d candidates, need >=2", len(candidate_images)
        )
        return None, {}

    with _log_timing("existing_pairs"):
        pair_set = _existing_pairs()

    with _log_timing("seed_pool"):
        seed_pool = set(_stable_seed_pool(all_images))
        seed_candidates = [
            img for img in candidate_images if img["filename"] in seed_pool
        ]

    reserve_count = int(config["ranking"]["reserve_count"])
    total_comps = get_total_comparisons()
    if total_comps % reserve_count == 0:
        _skip_before = 0

    # Phase 1: Seed Coverage (index 0)
    if _skip_before <= 0:
        with _log_timing("phase1_seed_coverage"):
            result = _phase1_seed_coverage(seed_candidates, pair_set)
        if result[0]:
            _skip_before = 0
            return result

    # Phase 2: Anchor Insert (index 1)
    if _skip_before <= 1:
        with _log_timing("phase2_anchor_insert"):
            result = _phase2_anchor_insert(candidate_images, seed_pool)
        if result[0]:
            _skip_before = 1
            return result

    # Phase 3: Collapsible Pairs (index 2)
    if _skip_before <= 2:
        with _log_timing("phase3_collapsible_pairs"):
            result = _phase3_collapsible_pairs(candidate_images, pair_set)
        if result[0]:
            _skip_before = 2
            return result

    # Phase 4: Chain Merge (index 3)
    if _skip_before <= 3:
        with _log_timing("phase4_chain_merge"):
            result = _phase4_chain_merge(candidate_images, pair_set)
        if result[0]:
            _skip_before = 3
            return result

    # Phase 5: Uncertainty Refine (index 4)
    if _skip_before <= 4:
        with _log_timing("phase5_uncertainty_refine"):
            result = _phase5_uncertainty_refine(candidate_images, pair_set)
        if result[0]:
            _skip_before = 4
            return result

    # Fallback (index 5)
    if _skip_before <= 5:
        with _log_timing("phase_fallback"):
            result = _phase_fallback(candidate_images, pair_set)
        if result[0]:
            _skip_before = 5
            return result

    logger.warning("select_pair: no pair found after all phases")
    return None, {}
