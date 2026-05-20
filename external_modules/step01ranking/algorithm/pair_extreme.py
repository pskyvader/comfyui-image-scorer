"""Loop 0.5 — Extreme pair selection from database records.

Pairs images that have ONLY ever won (pure winners) or ONLY ever lost
(pure losers), regardless of what the in-memory graph reflects.
"""

from typing import Any
import logging
import random

from database.comparisons_table import (
    get_images_with_only_wins,
    get_images_with_only_losses,
)
from .state import set_last_pair_metadata

logger = logging.getLogger(__name__)


def find_extreme_pair_from_db(
    candidate_filenames: set[str],
    comp_count_lookup: dict[str, int],
    only_wins: list[str],
    only_losses: list[str],
) -> tuple[str, str] | None:
    """Pair pure winners or pure losers queried directly from the database.

    Priority: pair same-extreme-type images with the least comparison counts first.
    """
    if len(only_wins) == 1 and len(only_losses) == 1:
        return None
    if len(only_losses) >= 2:
        only_losses.sort(key=lambda n: (comp_count_lookup[n], random.random()))
        pair = (only_losses[0], only_losses[1])
        set_last_pair_metadata(
            {
                "pair_type": "extreme_losses",
                "chain_level": -1,
                "component_size": 0,
                "left_comp_count": comp_count_lookup[pair[0]],
                "right_comp_count": comp_count_lookup[pair[1]],
            }
        )
        logger.info(
            f"[NEXT-PAIR] Loop=0.5 Type=pure_losers Pair={pair} "
            f"LeftComparisonCount={comp_count_lookup[pair[0]]} "
            f"RightComparisonCount={comp_count_lookup[pair[1]]}"
        )
        return pair

    if len(only_wins) >= 2:
        only_wins.sort(key=lambda n: (comp_count_lookup[n], random.random()))
        pair = (only_wins[0], only_wins[1])
        set_last_pair_metadata(
            {
                "pair_type": "extreme_wins",
                "chain_level": -1,
                "component_size": 0,
                "left_comp_count": comp_count_lookup[pair[0]],
                "right_comp_count": comp_count_lookup[pair[1]],
            }
        )
        logger.info(
            f"[NEXT-PAIR] Loop=0.5 Type=pure_winners Pair={pair} "
            f"LeftComparisonCount={comp_count_lookup[pair[0]]} "
            f"RightComparisonCount={comp_count_lookup[pair[1]]}"
        )
        return pair

    return None
