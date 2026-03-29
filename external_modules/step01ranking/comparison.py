"""
Comparison mode helpers for pair-based scoring refinement.

Two images with the same initial score are compared, and based on the winner,
a score_modifier is adjusted. When the modifier reaches ±5, the score is
adjusted by ±1 and the modifier resets. This allows fine-tuning of scores.
"""

from typing import Any
import math


from .utils import load_metadata, get_json_path
from shared.io import atomic_write_json
from .cache import add, get_image_pair


Meta = dict[str, int | float | str]
SeenItem = tuple[str, Meta, float]
Pair = tuple[str, str, Meta, Meta]

_cached_pairs: list[str] = []


def get_paired_images(
    score: int = 0,
    safety_limit: int = 20,
    max_comparison_count: int = 10,
    max_tolerance: float = 1.0,
) -> Pair | None:
    global _cached_pairs
    if len(_cached_pairs) > safety_limit:
        # remove oldest added elements
        _cached_pairs = _cached_pairs[-int(safety_limit / 2) :]
    image_pair = None
    i = None
    min_tolerance = tolerance = 0
    alpha = 2.0

    for i in range(1, max_comparison_count + 1):
        safety_limit_level = safety_limit * i
        tolerance = 0.01 + max_tolerance * (1 - (i / max_comparison_count) ** alpha)
        min_tolerance = tolerance
        while min_tolerance >= 0 and image_pair is None:
            min_tolerance -= 0.1
            image_pair: Pair | None = get_image_pair(
                i, tolerance, min_tolerance, score, safety_limit_level, _cached_pairs
            )

        if image_pair is not None:
            break
    if i:
        print(f"last level: {i}, tolerance: {min_tolerance} - {tolerance}")

    if image_pair is not None:
        _cached_pairs.append(image_pair[0])
        _cached_pairs.append(image_pair[1])
    return image_pair


# # 3️⃣ Choose level (Priority: Highest Level, Tie-breaker: Random Score)
# def choose_level(score: int = 0, safety_limit: int = 20, max_level:int=10) -> tuple[int, int, int]:
#     score_groups = get_total_per_level(score)
#     print(f"Score groups: {score_groups}")

#     # Store the highest valid level for each score: {score: highest_lvl}
#     best_per_score: Dict[int, int] = {}

#     for s, level_counts in score_groups.items():
#         highest_healthy_lvl = 0 if level_counts.get(0, 0) >= safety_limit else -1

#         for lvl in range(1, max_level):
#             current_count = level_counts.get(lvl, 0)
#             below_count = level_counts.get(lvl - 1, 0)

#             if current_count < safety_limit:
#                 continue

#             # Double Count Rule
#             if below_count < (2 * current_count):
#                 highest_healthy_lvl: int = lvl

#         if highest_healthy_lvl != -1:
#             best_per_score[s] = highest_healthy_lvl

#     if not best_per_score:
#         raise ValueError(f"No levels meet safety_limit ({safety_limit})")
#     print(f"best per score: {best_per_score}")

#     # Determine the absolute highest level reached by any score
#     min_level_found = min(best_per_score.values())

#     # Find all scores that reached this specific max level
#     top_candidates: List[tuple[int, int, int]] = []
#     for s, lvl in best_per_score.items():
#         if lvl == min_level_found:
#             count = score_groups[s][lvl]
#             top_candidates.append((s, lvl, count))

#     # top_candidates = [
#     #     (s, lvl) for s, lvl in best_per_score.items() if lvl == max_level_found
#     # ]
#     print(f"top_candidates: {top_candidates}")

#     # Return the only candidate, or pick randomly if there's a tie at the top level
#     return random.choice(top_candidates)


# def get_paired_images2(
#     score: int = 0,
#     safety_limit: int = 20,
#     tolerance: float = 0.001,
#     max_level: int=10
# ) -> Optional[Pair]:
#     global _cached_pairs
#     if len(_cached_pairs) > safety_limit:
#         # remove oldest added elements
#         _cached_pairs = _cached_pairs[-int(safety_limit / 2) :]

#     chosen_score: int
#     chosen_level: int
#     _: int

#     try:
#         chosen_score, chosen_level, _ = choose_level(score, safety_limit,max_level)
#     except ValueError:
#         return None

#     paths: List[str] = get_images_by_level(chosen_score, chosen_level)

#     if len(paths) < safety_limit:
#         return None

#     sampled_paths: List[str] = random.sample(paths, safety_limit)

#     seen: List[SeenItem] = []

#     best_pair: Optional[Pair] = None
#     smallest_diff: float = float("inf")

#     path: str
#     meta: Optional[Meta]
#     mod: float
#     diff: float
#     #print(f"sampled paths: {sampled_paths}")
#     for path in sampled_paths:
#         if path in _cached_pairs:
#             continue

#         meta = get_cached_metadata(path)
#         if meta is None:
#             continue

#         try:
#             mod = float(meta["score_modifier"])
#         except (KeyError, ValueError, TypeError):
#             continue

#         for path2, meta2, mod2 in seen:

#             diff = abs(mod - mod2)

#             if diff <= tolerance:
#                 # Immediate success
#                 _cached_pairs.append(path)
#                 _cached_pairs.append(path2)
#                 best_pair= path, path2, meta, meta2
#                 #print(f"best pair: {best_pair}")
#                 return best_pair

#             if diff < smallest_diff:
#                 smallest_diff = diff
#                 best_pair = (path, path2, meta, meta2)

#         seen.append((path, meta, mod))
#     if best_pair is not None and len(best_pair) >= 2:
#         _cached_pairs.append(best_pair[0])
#         _cached_pairs.append(best_pair[1])

#     #print(f"best pair: {best_pair}")

#     # Fallback: closest pair if no tolerance match
#     return best_pair


# Type for individual image data
DataDict = dict[str, Any]


def _apply_modifier(data: DataDict, delta: float) -> None:
    """Update score_modifier and comparison_count for a single entry."""
    data["score_modifier"] += delta
    data["comparison_count"] += 1


def _apply_extreme_cases(winner: DataDict, loser: DataDict) -> None:
    """Handle extreme score boundaries."""
    if winner["score"] >= 5 and winner["score_modifier"] > 0:
        loser["score_modifier"] -= winner["score_modifier"]
        winner["score_modifier"] = 0
    elif loser["score"] <= 1 and loser["score_modifier"] < 0:
        winner["score_modifier"] -= loser["score_modifier"]
        loser["score_modifier"] = 0


def _apply_threshold(data: DataDict, threshold: float = 5.5) -> None:
    """Adjust score based on threshold and reset score_modifier and comparison_count."""
    mod: float = data["score_modifier"]
    if mod > threshold:
        data["score"] += 1

    elif mod < -threshold:
        data["score"] -= 1
    if mod > threshold or mod < -threshold:
        data["score_modifier"] = 0
        data["comparison_count"] = 0
        data["volatility"] = 0


def write_comparison_data(
    file_path: str, data_update: DataDict
) -> tuple[bool, str | None]:
    """
    Update metadata and write to JSON for a single file.
    Returns (success, error_message)
    """
    meta: None | DataDict = load_metadata(file_path)
    if not meta:
        return False, "Could not load metadata"

    ts: str = next(iter(meta.keys()))
    meta[ts].update(data_update)

    json_path: str = get_json_path(file_path)
    print(f"Writing to {json_path}")
    atomic_write_json(json_path, meta, indent=4)

    # Update database / cache
    add(
        file_path,
        score=meta[ts]["score"],
        comparison_count=meta[ts]["comparison_count"],
        score_modifier=meta[ts]["score_modifier"],
        volatility=meta[ts]["volatility"],
    )

    return True, None


def _set_volatility(data: DataDict, delta: float) -> None:
    ALPHA_MOMENTUM: float = 0.30
    ALPHA_RESISTANCE: float = 0.13
    STABILITY_THRESHOLD: float = 0.5

    is_at_max = data["score"] == 5 and data["score_modifier"] >= 0
    is_at_min = data["score"] == 1 and data["score_modifier"] <= 0
    if data["comparison_count"] > 5 and (
        (is_at_max and delta == 1 and data["volatility"] > 0)
        or (is_at_min and delta == -1 and data["volatility"] < 0)
    ):
        delta = -delta
    alpha = ALPHA_MOMENTUM if (delta * data["volatility"] >= 0) else ALPHA_RESISTANCE
    data["volatility"] = (data["volatility"] * (1 - alpha)) + (delta * alpha)
    if data["comparison_count"] > 5 and abs(data["volatility"]) > STABILITY_THRESHOLD:
        data["comparison_count"] = 0

    data["volatility"] = max(-1.0, min(1.0, data["volatility"]))


# def _set_last_compared(data: DataDict, last_compared: str) -> None:
#     data["last_compared"] = last_compared
#     return None


# def _get_last_compared(path: str, data: DataDict) -> tuple[DataDict, str] | None:
#     metadata = get_cached_metadata(path)
#     if not metadata or not isinstance(metadata["last_compared"], str):
#         print(f"No last compared data found for {path}")
#         return None
#     last_compared_path = metadata["last_compared"]
#     last_compared = get_cached_metadata(last_compared_path)
#     if not last_compared or not last_compared["score"]:
#         print(f"No last compared data found for {path}")
#         return None
#     if int(last_compared["score"]) != int(data["score"]):
#         print(f"No last compared data found for {path}")
#         return None
#     return last_compared, last_compared_path


def apply_comparison_and_write(
    winner_data: DataDict,
    winner_path: str,
    loser_data: DataDict,
    loser_path: str,
) -> tuple[DataDict, DataDict, bool, str | None]:
    """
    Apply comparison, update score_modifiers, handle thresholds,
    and write both winner and loser to storage.
    """

    # 1️⃣ Compute the random delta
    score_scale = 0.5 + (2 * math.exp(-0.3 * winner_data["comparison_count"]))
    effective_score_winner = winner_data["score"] + winner_data["score_modifier"] / 10
    effective_score_loser = loser_data["score"] + loser_data["score_modifier"] / 10

    score_difference = effective_score_winner - effective_score_loser
    #if winner is below loser, then effectively switch places
    if score_difference < 0:
        score_scale = max(score_scale, -score_difference)

    # score_scale: float = 1 + 0.3 + random.random() * 0.2
    # last_compared_winner = _get_last_compared(winner_path, winner_data)
    # last_compared_loser = _get_last_compared(loser_path, loser_data)

    # 2️⃣ Apply modifiers
    _apply_modifier(winner_data, score_scale)
    _apply_modifier(loser_data, -score_scale)
    # if last_compared_winner:
    #     last_compared_winner[0]["score_modifier"] += score_scale / 2
    # if last_compared_loser:
    #     last_compared_loser[0]["score_modifier"] += score_scale / 2

    # 3️⃣ Handle extreme score boundaries
    _apply_extreme_cases(winner_data, loser_data)

    # 4️⃣ Apply thresholds
    _apply_threshold(winner_data)
    _apply_threshold(loser_data)
    _set_volatility(winner_data, 1)
    _set_volatility(loser_data, -1)

    # print(f"volatility, winner: {winner_data['volatility']}, loser:{loser_data['volatility']}")

    # if last_compared_winner:
    #     _apply_threshold(last_compared_winner[0])
    # if last_compared_loser:
    #     _apply_threshold(last_compared_loser[0])

    # _set_last_compared(winner_data, loser_path)
    # _set_last_compared(loser_data, winner_path)

    # 5️⃣ Write updated data
    winner_success: bool
    loser_success: bool
    winner_err: str | None
    loser_err: str | None

    winner_success, winner_err = write_comparison_data(winner_path, winner_data)
    if not winner_success:
        return winner_data, loser_data, False, f"Winner write failed: {winner_err}"

    loser_success, loser_err = write_comparison_data(loser_path, loser_data)
    if not loser_success:
        return winner_data, loser_data, False, f"Loser write failed: {loser_err}"

    # if last_compared_winner:
    #     _apply_threshold(last_compared_winner[0])
    #     _, _ = write_comparison_data(last_compared_winner[1], last_compared_winner[0])
    # if last_compared_loser:
    #     _, _ = write_comparison_data(last_compared_loser[1], last_compared_loser[0])

    return winner_data, loser_data, True, None
