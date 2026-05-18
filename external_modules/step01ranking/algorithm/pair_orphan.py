"""Loop 0 — Orphan pair selection.

Finds images with zero comparisons and pairs them together.
"""

from typing import Any
import logging
import random

from .state import set_last_pair_metadata

logger = logging.getLogger(__name__)


def find_orphan_pair(
    candidate_filenames: set[str], comp_count_lookup: dict[str, int]
) -> tuple[str, str] | None:
    """Pair two images that have never been compared (comparison count == 0)."""
    orphan_nodes: list[str] = [
        n for n in candidate_filenames if comp_count_lookup[n] == 0
    ]
    logger.debug(f"orphan nodes: {len(orphan_nodes)}")
    if len(orphan_nodes) >= 2:
        random.shuffle(orphan_nodes)
        pair = (orphan_nodes[0], orphan_nodes[1])
        set_last_pair_metadata({
            "pair_type": "orphan",
            "chain_level": -1,
            "component_size": 0,
            "left_comp_count": 0,
            "right_comp_count": 0,
        })
        logger.info(
            f"[NEXT-PAIR] Loop=0 Type=orphan Pair={pair} LeftComparisonCount=0 RightComparisonCount=0"
        )
        return pair
    return None
