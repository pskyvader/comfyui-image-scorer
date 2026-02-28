"""
Comparison mode helpers for pair-based scoring refinement.

Two images with the same initial score are compared, and based on the winner,
a score_modifier is adjusted. When the modifier reaches ±5, the score is
adjusted by ±1 and the modifier resets. This allows fine-tuning of scores.
"""

from typing import Tuple, Dict, Any, Optional, List
import random
from .utils import load_metadata, get_json_path
from shared.io import atomic_write_json
from .cache import add


def get_paired_images(
    manual_score: int = 0,
) -> Optional[Tuple[str, str, Dict[str, Any], Dict[str, Any]]]:
    """
    Get 2 random images with the same score for comparison.

    Conditions:
    - Both must have score == score (int 1-5)
    - comparison_count < 10 (not fully calibrated yet)

    Returns:
        Tuple of (img1_path, img2_path, img1_data, img2_data) or None if not available
    """
    from .cache import get_scored_not_compared, get_cached_metadata

    # Get all scored images that are not yet fully compared
    all_candidates:List[str]=[]
    for i in range(1,11):
        all_candidates = get_scored_not_compared(manual_score,i)
        if len(all_candidates) >=100:
            break
    print(
        f"[get_paired_images] Total scored not-compared images: {len(all_candidates)}"
    )
    if len(all_candidates) < 100:
        print("[get_paired_images] Safety function: Not enough candidates for pairing")
        return None
    random.shuffle(all_candidates)  # Shuffle to ensure random selection each time

    # Filter images with the given score and comparison_count < 10
    candidates: Dict[int, List[Tuple[str, Dict[str, Any]]]] = {}
    found_score = 0

    for img_path in all_candidates:
        cached_meta = get_cached_metadata(img_path)
        if not cached_meta:
            continue
        score = int(cached_meta["score"])
        if not score in candidates:
            candidates[score] = []
        meta = (img_path, cached_meta)
        candidates[score].append(meta)
        if len(candidates[score]) >= 2:
            if manual_score == 0 or manual_score == score:
                found_score = score
                break  # We only need at least 2 candidates for one score

    if candidates[found_score] and len(candidates[found_score]) >= 2:
        img1_path, img1_data = candidates[found_score][0]
        img2_path, img2_data = candidates[found_score][1]

        print(
            f"[get_paired_images]   Image 1: {img1_path} (count={img1_data.get('comparison_count', 0)})"
        )
        print(
            f"[get_paired_images]   Image 2: {img2_path} (count={img2_data.get('comparison_count', 0)})"
        )
        return (img1_path, img2_path, img1_data, img2_data)
    return None


def apply_comparison(
    winner_data: Dict[str, Any],
    loser_data: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Apply comparison result by updating score_modifier and potentially score.
    Returns:
        Tuple of (updated_winner_data, updated_loser_data)
    """
    print(f"compare BEFORE - Winner:{winner_data} - Loser:{loser_data}")
    if winner_data["score"] >= 5:
        loser_data["score_modifier"] -= 2
    elif loser_data["score"] <= 1:
        winner_data["score_modifier"] += 2
    else:
        winner_data["score_modifier"] += 1
        loser_data["score_modifier"] -= 1

    if winner_data["score_modifier"] > 5:
        winner_data["score"] += 1
        winner_data["score_modifier"] = 0
    elif winner_data["score_modifier"] < -5:
        winner_data["score"] -= 1
        winner_data["score_modifier"] = 0

    if loser_data["score_modifier"] > 5:
        loser_data["score"] += 1
        loser_data["score_modifier"] = 0
    elif loser_data["score_modifier"] < -5:
        loser_data["score"] -= 1
        loser_data["score_modifier"] = 0

    # Increment comparison count
    winner_data["comparison_count"] += 1
    loser_data["comparison_count"] += 1

    print(f"compared FINAL - Winner:{winner_data} - Loser:{loser_data}")

    return (winner_data, loser_data)



def write_comparison_data(
    winner_path: str,
    loser_path: str,
    winner_data: Dict[str, Any],
    loser_data: Dict[str, Any],
) -> Tuple[bool, Optional[str]]:
    """
    Write updated comparison data back to JSON files AND database.
    Also updates cache flags for fully_compared status.

    Returns:
        Tuple of (success: bool, error: Optional[str])
    """
    try:
        # Get metadata structure and update
        winner_meta = load_metadata(winner_path)
        loser_meta = load_metadata(loser_path)

        if not winner_meta or not loser_meta:
            return False, "Could not load metadata"

        # Get timestamp keys
        winner_ts = next(iter(winner_meta.keys()))
        loser_ts = next(iter(loser_meta.keys()))

        # Update metadata entries
        winner_meta[winner_ts].update(winner_data)
        loser_meta[loser_ts].update(loser_data)

        # Write back to JSON files (most important)
        winner_json_path = get_json_path(winner_path)
        loser_json_path = get_json_path(loser_path)

        atomic_write_json(winner_json_path, winner_meta, indent=4)
        atomic_write_json(loser_json_path, loser_meta, indent=4)
        add(
            winner_path,
            score=winner_data["score"],
            comparison_count=winner_data["comparison_count"],
            score_modifier=winner_data["score_modifier"],
        )

        add(
            loser_path,
            score=loser_data["score"],
            comparison_count=loser_data["comparison_count"],
            score_modifier=loser_data["score_modifier"],
        )

        return True, None
    except Exception as e:
        return False, str(e)
