"""Merge sort ranker - O(N log N) pair selection and scoring."""

from typing import Any
from database.images_table import (
    get_all_images,
    get_image as get_image_data,
    update_image_score,
    update_image_confidence,
)
from database.comparisons_table import (
    add_comparison,
)
from algorithm.confidence_tracker import calculate_confidence
from file_management.path_handler import (
    append_comparison_history_to_json,
)
from shared.config import config
from shared.graph import get_current_connections, ComparisonConnections
from datetime import datetime, timezone
import heapq
import math
import random
import time
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)
_graph_cache: dict[str, Any] = {"data": None, "timestamp": 0.0}
_GRAPH_CACHE_TTL = 30.5  # Cache graph for 5 seconds to reduce DB load

_images_cache: dict[str, Any] = {"data": None, "timestamp": 0.0}
_IMAGES_CACHE_TTL = 5.5  # Cache all_images for 2 seconds

_MAX_PAIR_CANDIDATES = 100


def _get_cached_all_images() -> list[dict[str, Any]]:
    """Return cached all_images list, refreshing if stale."""
    global _images_cache
    now = time.time()
    if (
        _images_cache["data"] is not None
        and (now - _images_cache["timestamp"]) < _IMAGES_CACHE_TTL
    ):
        return _images_cache["data"]

    data = get_all_images()
    _images_cache = {"data": data, "timestamp": now}
    return data


def _is_better_path(
    candidate_edges: int,
    candidate_impact: float,
    current: tuple[int, float] | None,
) -> bool:
    if current is None:
        return True

    current_edges, current_impact = current
    return candidate_edges < current_edges or (
        candidate_edges == current_edges and candidate_impact > current_impact
    )


def _collect_directional_paths(
    start: str, adjacency: defaultdict, max_edges: int
) -> dict[str, tuple[int, float]]:
    queue = deque([(start, 1.0, 0)])
    best_paths: dict[str, tuple[int, float]] = {start: (0, 1.0)}

    while queue:
        current, impact, edges = queue.popleft()
        if edges >= max_edges:
            continue

        for neighbor, edge_impact in adjacency.get(current, []):
            next_edges = edges + 1
            next_impact = impact * float(edge_impact)
            existing = best_paths.get(neighbor)
            if _is_better_path(next_edges, next_impact, existing):
                best_paths[neighbor] = (next_edges, next_impact)
                queue.append((neighbor, next_impact, next_edges))

    return best_paths


def _resolve_directional_indirect(
    winner: str,
    loser: str,
    winners_by_image: defaultdict,
    losers_by_image: defaultdict,
    max_edges: int,
    decay_factor: float,
) -> tuple[float, int] | None:
    winner_paths = _collect_directional_paths(winner, winners_by_image, max_edges)
    loser_paths = _collect_directional_paths(loser, losers_by_image, max_edges)

    best_match: tuple[int, float] | None = None

    for middle, (winner_edges, winner_impact) in winner_paths.items():
        if winner_edges == 0:
            continue

        loser_path = loser_paths.get(middle)
        if loser_path is None:
            continue

        loser_edges, loser_impact = loser_path
        if loser_edges == 0:
            continue

        total_edges = winner_edges + loser_edges
        if total_edges < 2 or total_edges > max_edges:
            continue

        transitive_depth = total_edges - 1
        impact_factor = winner_impact * loser_impact * (decay_factor**transitive_depth)

        if best_match is None:
            best_match = (transitive_depth, impact_factor)
            continue

        best_depth, best_impact = best_match
        if transitive_depth < best_depth or (
            transitive_depth == best_depth and impact_factor > best_impact
        ):
            best_match = (transitive_depth, impact_factor)

    if best_match is None:
        return None

    transitive_depth, impact_factor = best_match
    return impact_factor, transitive_depth


def _iter_chain_end_pairs(score_sorted_images: list[dict[str, Any]]):
    if len(score_sorted_images) < 2:
        return

    # We expect score_sorted_images to be sorted by score.

    last_idx = len(score_sorted_images) - 1
    heap: list[tuple[float, int, int]] = [
        (
            -abs(
                float(score_sorted_images[last_idx]["score"])
                - float(score_sorted_images[0]["score"])
            ),
            0,
            last_idx,
        )
    ]
    seen: set[tuple[int, int]] = {(0, last_idx)}

    while heap:
        _, left, right = heapq.heappop(heap)
        if left >= right:
            continue

        yield score_sorted_images[left], score_sorted_images[right]

        for next_left, next_right in ((left + 1, right), (left, right - 1)):
            if next_left >= next_right or (next_left, next_right) in seen:
                continue

            seen.add((next_left, next_right))
            diff = abs(
                float(score_sorted_images[next_right]["score"])
                - float(score_sorted_images[next_left]["score"])
            )
            heapq.heappush(heap, (-diff, next_left, next_right))


def _randomize_pair_order(filename_a: str, filename_b: str) -> tuple[str, str]:
    if random.random() < 0.5:
        return filename_a, filename_b
    return filename_b, filename_a


def _get_allowed_diff_for_image(image: dict) -> float:
    return max(0.05, 1.0 * math.exp(-0.3 * int(image["comparison_count"])))


def _get_allowed_diff_for_pair(image_a: dict, image_b: dict) -> float:
    return max(
        _get_allowed_diff_for_image(image_a),
        _get_allowed_diff_for_image(image_b),
    )


def _search_pairs_by_score_gap(
    score_sorted_images: list[dict[str, Any]],
    compared_pairs: set[tuple[str, str]],
    winners_by_image: defaultdict,
    losers_by_image: defaultdict,
) -> tuple[str, str] | None:
    current_diff: float = -1
    explicit_candidates: list[tuple[str, str]] = []
    transitive_record_count = 0

    for img_low, img_high in _iter_chain_end_pairs(score_sorted_images):
        filename_a = img_low["filename"]
        filename_b = img_high["filename"]
        pair_key = tuple(sorted((filename_a, filename_b)))

        if pair_key in compared_pairs:
            logger.debug(f"[SEARCH_PAIRS] Pair {pair_key} already compared")
            continue

        score_diff = abs(float(img_high["score"]) - float(img_low["score"]))
        allowed_diff = _get_allowed_diff_for_pair(img_low, img_high)

        if score_diff > allowed_diff:
            logger.debug(
                f"[SEARCH_PAIRS] Pair {pair_key} has score diff {score_diff} > {allowed_diff}"
            )
            continue

        if score_diff > current_diff:
            current_diff = score_diff

        winner, impact_factor, transitive_depth = check_indirect(
            filename_a,
            filename_b,
            winners_by_image,
            losers_by_image,
        )
        if winner:
            logger.debug(
                f"[NEXT-PAIR] Found indirect win for: {winner} ({transitive_depth})"
            )
            if record_comparison(
                filename_a,
                filename_b,
                winner,
                impact_factor=impact_factor,
                transitive_depth=transitive_depth,
            ):
                compared_pairs.add(pair_key)
                transitive_record_count += 1
                if transitive_record_count >= 10:
                    logger.debug("[SEARCH_PAIRS] Reached transitive recording limit (10)")
                    continue
                continue
            raise RuntimeError(
                f"Failed to record comparison for {filename_a} vs {filename_b}"
            )

        # Greedily pick the first valid candidate (which has the largest diff due to heap sorting)
        logger.debug(f"[SEARCH_PAIRS] Greedily picked largest diff: {filename_a} vs {filename_b} (diff: {score_diff})")
        return _randomize_pair_order(filename_a, filename_b)

    return None


def _search_confidence_pool(
    score_sorted_images: list[dict[str, Any]],
    compared_pairs: set[tuple[str, str]],
    winners_by_image: defaultdict,
    losers_by_image: defaultdict,
    component_by_filename: dict[str, int],
    chain_length_by_filename: dict[str, int],
) -> tuple[str, str] | None:
    active_chains: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for image in score_sorted_images:
        active_chains[component_by_filename[image["filename"]]].append(image)

    chain_groups = [
        {
            "component_id": component_id,
            "members": members,
            "chain_length": chain_length_by_filename[members[0]["filename"]],
        }
        for component_id, members in active_chains.items()
    ]
    groups_by_length: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for chain_group in chain_groups:
        groups_by_length[int(chain_group["chain_length"])].append(chain_group)

    sorted_lengths = sorted(groups_by_length)
    for idx_a, length_a in enumerate(sorted_lengths):
        groups_a = groups_by_length[length_a]

        length_pairs: list[tuple[int, int]] = []
        if len(groups_a) >= 2:
            length_pairs.append((length_a, length_a))
        for length_b in sorted_lengths[idx_a + 1 :]:
            length_pairs.append((length_a, length_b))

        for pair_length_a, pair_length_b in length_pairs:
            candidate_groups_a = groups_by_length[pair_length_a]
            candidate_groups_b = groups_by_length[pair_length_b]

            if pair_length_a == pair_length_b and all(
                len(group["members"]) == 1 for group in candidate_groups_a
            ):
                logger.debug("[NEXT-PAIR] Found singletons")
                singleton_images = sorted(
                    (group["members"][0] for group in candidate_groups_a),
                    key=lambda image: float(image["score"]),
                )

                # getSingleton images unordered
                # singleton_images = list(
                #     group["members"][0] for group in candidate_groups_a
                # )
                # random.shuffle(singleton_images)

                pair = _search_pairs_by_score_gap(
                    singleton_images,
                    compared_pairs,
                    winners_by_image,
                    losers_by_image,
                )
                if pair:
                    return pair
                continue

            current_diff: float = -1.0
            explicit_candidates: list[tuple[str, str]] = []

            for group_idx_a, chain_a in enumerate(candidate_groups_a):
                start_b = group_idx_a + 1 if pair_length_a == pair_length_b else 0
                for group_idx_b in range(start_b, len(candidate_groups_b)):
                    chain_b = candidate_groups_b[group_idx_b]

                    pair_options = [
                        # opposite extremes: a_high vs b_low
                        (chain_a["members"][0], chain_b["members"][-1]),
                        (chain_a["members"][-1], chain_b["members"][0]),
                        # same extremes:  a_high vs b_high
                        (chain_a["members"][0], chain_b["members"][0]),
                        (chain_a["members"][-1], chain_b["members"][-1]),
                    ]
                    option_diffs = [
                        abs(float(high_end["score"]) - float(low_end["score"]))
                        for low_end, high_end in pair_options
                    ]
                    score_diff = max(option_diffs)

                    if score_diff > current_diff:
                        current_diff = score_diff

                    for option_idx, (low_end, high_end) in enumerate(pair_options):
                        if option_diffs[option_idx] != score_diff:
                            continue

                    for option_idx, (low_end, high_end) in enumerate(pair_options):
                        score_diff = option_diffs[option_idx]
                        filename_a = low_end["filename"]
                        filename_b = high_end["filename"]
                        pair_key = tuple(sorted((filename_a, filename_b)))
                        allowed_diff = _get_allowed_diff_for_pair(low_end, high_end)

                        if pair_key in compared_pairs or score_diff > allowed_diff:
                            continue

                        winner, impact_factor, transitive_depth = check_indirect(
                            filename_a,
                            filename_b,
                            winners_by_image,
                            losers_by_image,
                        )
                        if winner:
                            raise RuntimeError(
                                "Link found when comparing different chains. Chain creation algorithm failing."
                            )
                        
                        explicit_candidates.append((score_diff, filename_a, filename_b))

            if explicit_candidates:
                # Sort by score difference descending and pick the best
                explicit_candidates.sort(key=lambda x: x[0], reverse=True)
                _, chosen_a, chosen_b = explicit_candidates[0]
                logger.debug(f"[SEARCH_CONFIDENCE] Picked best inter-chain pair: {chosen_a} vs {chosen_b} (diff: {explicit_candidates[0][0]})")
                return _randomize_pair_order(chosen_a, chosen_b)

    return _search_pairs_by_score_gap(
        score_sorted_images,
        compared_pairs,
        winners_by_image,
        losers_by_image,
    )


def check_indirect(
    a: str,
    b: str,
    winners_by_image: defaultdict,
    losers_by_image: defaultdict,
) -> tuple[str | None, float, int]:
    if a in losers_by_image and b in losers_by_image[a]:
        return a, 1.0, 0
    if b in losers_by_image and a in losers_by_image[b]:
        return b, 1.0, 0

    ranking_config = config.get("ranking")
    max_edges = int(ranking_config.get("transitive_depth"))
    decay_factor = float(ranking_config.get("indirect_impact_decay"))

    result_a = _resolve_directional_indirect(
        a,
        b,
        winners_by_image,
        losers_by_image,
        max_edges,
        decay_factor,
    )
    if result_a is not None:
        impact_factor, transitive_depth = result_a
        return a, impact_factor, transitive_depth

    result_b = _resolve_directional_indirect(
        b,
        a,
        winners_by_image,
        losers_by_image,
        max_edges,
        decay_factor,
    )
    if result_b is not None:
        impact_factor, transitive_depth = result_b
        return b, impact_factor, transitive_depth
    return None, 0.0, 0


def select_pair_for_comparison(
    exclude_set: set[str] | None = None,
) -> tuple[str, str] | None:
    """
    Select the next pair of images to compare.

    Linear pipeline:
    1. Get all images from database (cached)
    2. Filter out recently shown images (exclude_set)
    3. Get comparison graph data (cached)
    4. Find lowest confidence images
    5. Try chain-end pairs, then transitive resolution, then fallback
    """
    total_time = time.time()
    step_time = time.time()
    all_images = _get_cached_all_images()
    logger.debug(
        f"[NEXT-PAIR] all images: {len(all_images)},  time: total: {time.time() - total_time}, step: {time.time() - step_time}"
    )
    step_time = time.time()

    if len(all_images) < 2:
        return None

    candidate_images = _filter_excluded_images(all_images, exclude_set)
    logger.debug(
        f"[NEXT-PAIR] filter_excluded_images:{len(candidate_images)} time: total: {time.time() - total_time}, step: {time.time() - step_time}"
    )
    step_time = time.time()

    if len(candidate_images) < 2:
        return None

    graph_data = _get_comparison_graph(all_images)
    logger.debug(
        f"[NEXT-PAIR] get_comparison_graph:{len(graph_data.all_filenames)} time: total: {time.time() - total_time}, step: {time.time() - step_time}"
    )
    step_time = time.time()

    low_conf_images = _find_lowest_confidence_images(candidate_images)
    logger.debug(
        f"[NEXT-PAIR] find_lowest_confidence_images:{len(low_conf_images)} time: total: {time.time() - total_time}, step: {time.time() - step_time}"
    )
    step_time = time.time()

    pair = _find_pair_from_confidence_pool(low_conf_images, graph_data)
    logger.debug(
        f"[NEXT-PAIR] find_pair_from_confidence_pool:{pair} time: total: {time.time() - total_time}, step: {time.time() - step_time}"
    )
    step_time = time.time()
    if pair:
        return pair

    pair = _find_fallback_pair(low_conf_images, graph_data.compared_pairs)
    if not pair:
         pair = _find_fallback_pair(candidate_images, graph_data.compared_pairs)
    logger.debug(
        f"[NEXT-PAIR] find_fallback_pair:{pair} time: total: {time.time() - total_time}, step: {time.time() - step_time}"
    )
    return pair


def _filter_excluded_images(
    images: list[dict[str, Any]], exclude_set: set[str] | None
) -> list[dict[str, Any]]:
    """Step 2: Filter out recently shown images."""
    if not exclude_set or len(images) <= 5:
        return images

    max_exclude = int(len(images) * 0.75)
    safe_exclude = (
        set(list(exclude_set)[:max_exclude])
        if len(exclude_set) > max_exclude
        else exclude_set
    )
    return [img for img in images if img["filename"] not in safe_exclude]


def _get_comparison_graph(all_images: list[dict[str, Any]]) -> ComparisonConnections:
    """Step 3: Get comparison graph data needed for filtering. Uses caching."""
    global _graph_cache

    t0 = time.time()
    if (
        _graph_cache["data"] is not None
        and (t0 - _graph_cache["timestamp"]) < _GRAPH_CACHE_TTL
    ):
        return _graph_cache["data"]

    result = get_current_connections(
        all_images=all_images,
        include_transitive=False,
    )
    _graph_cache = {"data": result, "timestamp": t0}
    return result


def _find_lowest_confidence_images(
    images: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Step 4: Find images for comparison.

    Random selection from ends + middle for score diversity, limited to _MAX_PAIR_CANDIDATES.
    """
    if not images:
        return []

    # filter by confidence - allow a small margin (0.05) to include images in the same tier
    confidence = min(img["confidence"] for img in images)
    images = [img for img in images if img["confidence"] <= confidence + 0.05]
    # Use a random tie-breaker for score sorting to avoid filename bias
    # and ensure stratified sampling (ends/middle) picks from different parts of the pool.
    random_tiebreakers = {img["filename"]: random.random() for img in images}
    sorted_by_score = sorted(
        images,
        key=lambda img: (float(img["score"]), random_tiebreakers[img["filename"]]),
    )

    n = len(sorted_by_score)
    if n <= _MAX_PAIR_CANDIDATES:
        return sorted_by_score

    candidates = list(sorted_by_score)
    section_size = math.floor(_MAX_PAIR_CANDIDATES / 4)

    # candidates= first section + middle section + last section
    candidates = (
        candidates[:section_size]
        + candidates[(n // 2 - section_size) : (n // 2 + section_size + 1)]
        + candidates[n - section_size :]
    )
    # make sure the candidates are not duplicated, based on filename
    unique_candidates: list[dict[str, Any]] = []
    for img in candidates:
        if img["filename"] not in [img["filename"] for img in unique_candidates]:
            unique_candidates.append(img)

    random.shuffle(unique_candidates)

    return unique_candidates[:_MAX_PAIR_CANDIDATES]


def _find_pair_from_confidence_pool(
    images: list[dict[str, Any]], graph_data: Any
) -> tuple[str, str] | None:
    """Step 5a: Try confidence-based pair selection."""
    if not images:
        return None

    # We MUST sort by score here because _search_confidence_pool and _iter_chain_end_pairs
    # expect a score-sorted list to find the "extremes" (min/max scores).
    # Shuffling was done earlier for candidate pool diversity, but now we need order.
    score_sorted = sorted(
        images,
        key=lambda img: (float(img["score"]), img["filename"]),
    )

    compared_pairs: set[tuple[str, str]] = (
        set(graph_data.compared_pairs)
        if hasattr(graph_data, "compared_pairs")
        else set()
    )

    pair = _search_confidence_pool(
        score_sorted,
        compared_pairs,
        graph_data.winners_by_image,
        graph_data.losers_by_image,
        graph_data.component_by_filename,
        graph_data.chain_length_by_filename,
    )
    return pair


def _find_fallback_pair(
    images: list[dict[str, Any]], compared_pairs: set[tuple[str, str]]
) -> tuple[str, str] | None:
    """Step 5b: Fallback - return first uncompared pair."""
    if not images:
        return None

    n = len(images)
    # Sort by confidence first, but shuffle within confidence levels to avoid filename bias.
    # We do this by shuffling first, then performing a stable sort by confidence.
    shuffled_images = list(images)
    random.shuffle(shuffled_images)
    sorted_images = sorted(shuffled_images, key=lambda img: img["confidence"])

    for i in range(n):
        for j in range(i + 1, n):
            a = sorted_images[i]["filename"]
            b = sorted_images[j]["filename"]
            if tuple(sorted((a, b))) in compared_pairs:
                continue
            return (a, b)
    return None


def update_scores_after_comparison(
    winner_filename: str,
    loser_filename: str,
    winner_data: dict,
    loser_data: dict,
    impact_factor: float = 1.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Update scores for both images after a comparison."""
    winner_score = winner_data.get("score", 0.5)
    loser_score = loser_data.get("score", 0.5)
    winner_comp_count = winner_data.get("comparison_count", 0)
    loser_comp_count = loser_data.get("comparison_count", 0)

    # New "Fast Jump" formula: 0.5 at 0 comps, 0.05 at 10 comps
    winner_delta = 0.5 * math.exp(-0.23 * winner_comp_count) * impact_factor
    loser_delta = 0.5 * math.exp(-0.23 * loser_comp_count) * impact_factor

    new_winner_score = max(0.0, min(1.0, winner_score + winner_delta))
    new_loser_score = max(0.0, min(1.0, loser_score - loser_delta))

    winner_data["score"] = new_winner_score
    winner_data["comparison_count"] = winner_comp_count + 1
    loser_data["score"] = new_loser_score
    loser_data["comparison_count"] = loser_comp_count + 1

    return winner_data, loser_data


def record_comparison(
    filename_a: str,
    filename_b: str,
    winner: str,
    impact_factor: float = 1.0,
    transitive_depth: int = 0,
) -> bool:
    """Record a comparison and update image scores/confidence."""
    data_a = get_image_data(filename_a)
    data_b = get_image_data(filename_b)

    if not data_a or not data_b or filename_a == filename_b:
        return False

    if winner == filename_a:
        winner_data, loser_data = data_a, data_b
        winner_filename, loser_filename = filename_a, filename_b
    else:
        winner_data, loser_data = data_b, data_a
        winner_filename, loser_filename = filename_b, filename_a

    winner_data, loser_data = update_scores_after_comparison(
        winner_filename, loser_filename, winner_data, loser_data, impact_factor
    )

    ts = datetime.now(timezone.utc).isoformat()
    comp_id = add_comparison(
        filename_a,
        filename_b,
        winner,
        impact_factor,
        transitive_depth,
        timestamp=ts,
    )
    if not comp_id:
        return False

    update_image_score(winner_filename, winner_data["score"])
    update_image_confidence(
        winner_filename,
        calculate_confidence(
            winner_filename, winner_data["score"], winner_data["comparison_count"]
        ),
        winner_data["comparison_count"],
    )
    update_image_score(loser_filename, loser_data["score"])
    update_image_confidence(
        loser_filename,
        calculate_confidence(
            loser_filename, loser_data["score"], loser_data["comparison_count"]
        ),
        loser_data["comparison_count"],
    )

    entry_winner = {
        "comparison_id": comp_id,
        "other": loser_filename,
        "winner": True,
        "weight": impact_factor,
        "transitive_depth": transitive_depth,
        "timestamp": ts,
    }
    entry_loser = {
        "comparison_id": comp_id,
        "other": winner_filename,
        "winner": False,
        "weight": impact_factor,
        "transitive_depth": transitive_depth,
        "timestamp": ts,
    }

    winner_json_saved = append_comparison_history_to_json(
        winner_filename,
        entry_winner,
        new_score=winner_data["score"],
        new_confidence=calculate_confidence(
            winner_filename,
            winner_data["score"],
            winner_data["comparison_count"],
        ),
    )
    loser_json_saved = append_comparison_history_to_json(
        loser_filename,
        entry_loser,
        new_score=loser_data["score"],
        new_confidence=calculate_confidence(
            loser_filename,
            loser_data["score"],
            loser_data["comparison_count"],
        ),
    )

    if not winner_json_saved or not loser_json_saved:
        raise RuntimeError(
            f"Comparison history save incomplete for comp_id={comp_id} "
            f"{winner_filename}<->{loser_filename}: winner_saved={winner_json_saved}, loser_saved={loser_json_saved}"
        )

    # Invalidate caches after recording
    _graph_cache["data"] = None
    _images_cache["data"] = None

    return True
