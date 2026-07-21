"""Phase ordering configuration.

The single source of truth for which phases exist, in which order they run,
and the metadata (labels, CSS classes, descriptions, conditional flags) that
the frontend uses to render them.  Reorder the PHASES list to change the
execution order — the list index IS the phase number.
"""

from __future__ import annotations

import random
import time
from typing import Any

from ....shared.logger import SharedLogger
from ....shared.config import config
from ...database_structure.comparisons_table import get_total_comparisons

from .constants import MIN_CHAIN_THRESHOLD
from .pair_active import (
    _phase1_seed_coverage,
    _phase2_anchor_insert,
    _phase3_collapsible_pairs,
    _phase4_chain_merge,
    _phase5_uncertainty_refine,
    _phase_fallback,
    _existing_pairs,
    _stable_seed_pool,
)

logger = SharedLogger.get_logger(__name__)


# Each entry is a dict whose first key is the phase name (mapping to its
# function).  Remaining keys are metadata consumed by the frontend.
PHASES: list[dict[str, Any]] = [
    {
        "seed": _phase1_seed_coverage,
        "phase_label": "Phase 1 / Bootstrap Seed",
        "card_class": "card-bootstrap",
        "description_class": "text-purple-400",
        "description": "ensures seed images (top {seed_size} by comparisons) reach {seed_target} comparisons each, preferring cross-path opponents with similar scores",
        "show_chain_info": False,
        "show_mu_sigma": False,
    },
    {
        "anchor": _phase2_anchor_insert,
        "phase_label": "Phase 2 / Anchor Insert",
        "card_class": "card-anchor",
        "description_class": "text-blue-400",
        "description": "integrates new images with \u2264{insertion_target} comparisons by pairing them with the closest mu (skill) from a different crystal path, preferring different components",
        "show_chain_info": False,
        "show_mu_sigma": False,
    },
    {
        "collapsible": _phase3_collapsible_pairs,
        "phase_label": "Phase 3 / Collapsible",
        "card_class": "card-collapsible",
        "description_class": "text-emerald-400",
        "description": "finds two tops or two bottoms in the same component not yet transitively connected; one click resolves ranking for entire branches",
        "show_chain_info": True,
        "show_mu_sigma": False,
    },
    {
        "refine": _phase5_uncertainty_refine,
        "phase_label": "Phase 4 / Uncertainty Refine",
        "card_class": "card-refine",
        "description_class": "text-amber-400",
        "description": "reduces uncertainty by comparing highest-sigma images against the closest-mu seed images",
        "show_chain_info": False,
        "show_mu_sigma": False,
    },
    {
        "chain_merge": _phase4_chain_merge,
        "phase_label": "Phase 5 / Chain Merge",
        "card_class": "card-chain-merge",
        "description_class": "text-red-400",
        "description": "merges the longest chains by comparing internal mid-chain nodes, reducing the total number of chains",
        "show_chain_info": True,
        "show_mu_sigma": True,
    },
    {
        "fallback": _phase_fallback,
        "phase_label": "Fallback",
        "card_class": "card-fallback",
        "description_class": "text-purple-300/60",
        "description": "last-resort scan for any unseen pair when all heuristics fail",
        "show_chain_info": False,
        "show_mu_sigma": False,
    },
]


def get_phases() -> list[dict[str, Any]]:
    """Return a JSON-serializable version of PHASES (callables stripped).

    Each output dict includes the ``name`` key (derived from the callable
    key in the source entry) for the frontend to identify the phase.
    """
    result: list[dict[str, Any]] = []
    for entry in PHASES:
        cleaned: dict[str, Any] = {}
        name: str | None = None
        for k, v in entry.items():
            if callable(v):
                name = k
            else:
                cleaned[k] = v
        cleaned["name"] = name
        result.append(cleaned)
    return result


_skip_before: int = 0


def reset_skip() -> None:
    global _skip_before
    _skip_before = 0


def select_pair(
    all_images: list[dict[str, Any]],
    candidate_images: list[dict[str, Any]],
) -> tuple[tuple[str, str] | None, int | None]:
    global _skip_before

    if len(candidate_images) < 2:
        logger.warning(
            "select_pair: only %d candidates, need >=2", len(candidate_images)
        )
        return None, None

    existing_pairs_set = _existing_pairs()

    seed_pool = set(_stable_seed_pool(all_images))

    random.shuffle(candidate_images)

    seed_candidates = [img for img in candidate_images if img["filename"] in seed_pool]

    reserve_count = int(config["ranking"]["reserve_count"])
    total_comps: int = get_total_comparisons()
    logger.debug(f"total comps: {total_comps}, skip before:{_skip_before}")
    if total_comps % reserve_count == 0:
        reset_skip()

    for idx, phase in enumerate(PHASES):
        name = next(k for k, v in phase.items() if callable(v))
        fn = phase[name]

        if _skip_before > idx:
            continue

        if name == "seed":
            result = fn(seed_candidates, existing_pairs_set)
        elif name == "anchor":
            result = fn(candidate_images, seed_pool, existing_pairs_set)
        elif name == "collapsible":
            result = fn(candidate_images, existing_pairs_set)
        elif name == "refine":
            result = fn(candidate_images, existing_pairs_set)
        elif name == "chain_merge":
            result = fn(candidate_images)
        elif name == "fallback":
            result = fn(candidate_images, existing_pairs_set)
        else:
            result = None

        if result:
            _skip_before = idx
            return result, idx

    logger.warning("select_pair: no pair found after all phases")
    return None, None
