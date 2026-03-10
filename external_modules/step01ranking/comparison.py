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
from .cache import add, get_total_per_level, get_images_by_level, get_cached_metadata


# 3️⃣ Choose level (Priority: Highest Level, Tie-breaker: Random Score)
def choose_level(score: int = 0, safety_limit: int = 20) -> Tuple[int, int, int]:
    score_groups = get_total_per_level(score)
    print(f"Score groups: {score_groups}")

    # Store the highest valid level for each score: {score: highest_lvl}
    best_per_score: Dict[int, int] = {}

    for s, level_counts in score_groups.items():
        highest_healthy_lvl = 0 if level_counts.get(0, 0) >= safety_limit else -1

        for lvl in range(1, 10):
            current_count = level_counts.get(lvl, 0)
            below_count = level_counts.get(lvl - 1, 0)

            if current_count < safety_limit:
                continue

            # Double Count Rule
            if below_count < (2 * current_count):
                highest_healthy_lvl = lvl

        if highest_healthy_lvl != -1:
            best_per_score[s] = highest_healthy_lvl

    if not best_per_score:
        raise ValueError(f"No levels meet safety_limit ({safety_limit})")
    print(f"best per score: {best_per_score}")

    # Determine the absolute highest level reached by any score
    min_level_found = min(best_per_score.values())

    # Find all scores that reached this specific max level
    top_candidates: List[Tuple[int, int, int]] = []
    for s, lvl in best_per_score.items():
        if lvl == min_level_found:
            count = score_groups[s][lvl]
            top_candidates.append((s, lvl, count))

    # top_candidates = [
    #     (s, lvl) for s, lvl in best_per_score.items() if lvl == max_level_found
    # ]
    print(f"top_candidates: {top_candidates}")

    # Return the only candidate, or pick randomly if there's a tie at the top level
    return random.choice(top_candidates)


# 4️⃣ Get paired images
def get_paired_images(score: int = 0, safety_limit: int = 20, tolerance: float = 0.001):

    try:
        # Now prioritizes the deepest part of the pyramid
        chosen_score, chosen_level, _ = choose_level(score, safety_limit)
    except ValueError:
        return None

    paths = get_images_by_level(chosen_score, chosen_level)
    if len(paths) < 2:
        return None

    random.shuffle(paths)

    anchor_path = paths.pop()
    anchor_meta = get_cached_metadata(anchor_path)
    if not anchor_meta:
        return None

    anchor_mod = float(anchor_meta["score_modifier"])
    best_match = None
    smallest_diff = float("inf")

    # Greedy search capped at 50 to maintain performance
    search_limit = min(len(paths), 50)
    for path in paths[:search_limit]:
        meta = get_cached_metadata(path)
        if not meta:
            continue

        diff = abs(float(meta["score_modifier"]) - anchor_mod)

        if diff < smallest_diff:
            smallest_diff = diff
            best_match = (path, meta)

        if diff <= tolerance:
            print("found pair with tolerance <=", tolerance, diff)
            break

    print(f"smallest diff: {smallest_diff}")

    if best_match:
        return anchor_path, best_match[0], anchor_meta, best_match[1]

    return None


def get_paired_images2(
    manual_score: int = 0, safety_limit: int = 20
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
    all_candidates: List[str] = []
    for i in range(1, 11):
        all_candidates = get_scored_not_compared(manual_score, i)
        if len(all_candidates) >= safety_limit:
            break
    print(f"Total not-compared images: {len(all_candidates)} for count < {i}")
    if len(all_candidates) < safety_limit:
        print("[get_paired_images] Safety function: Not enough candidates for pairing")
        return None
    random.shuffle(all_candidates)  # Shuffle to ensure random selection each time

    # Filter images with the given score and comparison_count < 10
    candidates: Dict[int, Dict[float, List[Tuple[str, Dict[str, Any]]]]] = {}
    found_score = -1
    found_modifier = -1

    backup_pair: List[Tuple[str, Dict[str, Any]]] = []

    for img_path in all_candidates:
        cached_meta = get_cached_metadata(img_path)
        if not cached_meta:
            continue
        score = int(cached_meta["score"])
        score_modifier = round(float(cached_meta["score_modifier"]), 1)
        if not score in candidates:
            candidates[score] = {}
        if not score_modifier in candidates[score]:
            candidates[score][score_modifier] = []
        meta = (img_path, cached_meta)

        if len(backup_pair) < 2:
            backup_pair.append(meta)

        candidates[score][score_modifier].append(meta)
        if len(candidates[score][score_modifier]) >= 2:
            if manual_score == 0 or manual_score == score:
                found_score = score
                found_modifier = score_modifier
                break  # We only need at least 2 candidates for one score
    if found_score == -1:
        candidates[found_score] = {}
        candidates[found_score][found_modifier] = backup_pair

    if (
        candidates[found_score][found_modifier]
        and len(candidates[found_score][found_modifier]) >= 2
    ):
        img1_path, img1_data = candidates[found_score][found_modifier][0]
        img2_path, img2_data = candidates[found_score][found_modifier][1]

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
    score_scale = 1 + 0.3 + random.random() * 0.2
    winner_data["score_modifier"] += score_scale
    loser_data["score_modifier"] -= score_scale

    # Increment comparison count
    winner_data["comparison_count"] += 1
    loser_data["comparison_count"] += 1

    if winner_data["score"] >= 5 and winner_data["score_modifier"] > 0:
        loser_data["score_modifier"] -= winner_data["score_modifier"]
        winner_data["score_modifier"] = 0
    elif loser_data["score"] <= 1 and loser_data["score_modifier"] < 0:
        winner_data["score_modifier"] -= loser_data["score_modifier"]
        loser_data["score_modifier"] = 0

    # Score level change
    threshold = 5.5
    if (
        winner_data["score_modifier"] > threshold
        or winner_data["score_modifier"] < -threshold
    ):
        if winner_data["score_modifier"] > threshold:
            winner_data["score"] += 1
        else:
            winner_data["score"] -= 1
        winner_data["score_modifier"] = 0
        winner_data["comparison_count"] = 0

    if (
        loser_data["score_modifier"] > threshold
        or loser_data["score_modifier"] < -threshold
    ):
        if loser_data["score_modifier"] > threshold:
            loser_data["score"] += 1
        else:
            loser_data["score"] -= 1
        loser_data["score_modifier"] = 0
        loser_data["comparison_count"] = 0

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
