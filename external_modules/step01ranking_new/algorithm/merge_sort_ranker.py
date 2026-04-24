"""Merge sort ranker - O(N log N) pair selection and scoring."""

from typing import Optional, Tuple, List
from database.images_table import (
    get_all_images,
    get_image as get_image_data,
    update_image_score,
    update_image_confidence,
)
from database.comparisons_table import get_comparison_count, add_comparison, get_all_comparisons
from algorithm.confidence_tracker import calculate_confidence
from file_management.path_handler import append_comparison_history_to_json
from shared.config import config
from datetime import datetime
import math
import random
import bisect
from collections import defaultdict, deque


def bfs(start, end,graph):
    queue = deque([(start, 1.0, 0)])
    visited = set([start])
    max_depth = int(config.get("ranking").get("transitive_depth"))
    
    while queue:
        curr, imp, depth = queue.popleft()
        if curr == end: return imp, depth
        if depth >= max_depth: continue
        for neighbor, edge_imp in graph[curr]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, imp * edge_imp, depth + 1))
    return None, None

def check_indirect(A: str, B: str,graph:defaultdict) -> Tuple[Optional[str], float, int]:
    
    imp_A, depth_A = bfs(A, B,graph)
    imp_B, depth_B = bfs(B, A,graph)
    if imp_A is not None and imp_B is not None: return None, 0.0, 0
    if imp_A is not None: return A, imp_A, depth_A
    if imp_B is not None: return B, imp_B, depth_B
    return None, 0.0, 0


def select_pair_for_comparison(exclude_set: set = None) -> Optional[Tuple[str, str]]:
    """
    Select next pair of images to compare using a "Random Seed + Max Distance" strategy.
    
    Rules implemented:
    1. Filter out recently shown images (exclude_set).
    2. Identify the lowest comparison count in the library.
    3. Pick a random "Seed" image A from those with the lowest count.
    4. Use binary search to pick a "Partner" image B that is as far apart as possible from A within range.
    5. Automatically resolve indirect paths.
    """
    all_images = get_all_images()
    
    if len(all_images) < 2:
        return None

    # Filter out excluded images
    if exclude_set and len(all_images) > 5:
        # Safety: Don't exclude more than 75% of images, otherwise we might run out of pairs
        max_exclude = int(len(all_images) * 0.75)
        safe_exclude = set(list(exclude_set)[:max_exclude]) if len(exclude_set) > max_exclude else exclude_set
        all_images = [img for img in all_images if img["filename"] not in safe_exclude]

    if len(all_images) < 2:
        return None

    
    # 2. Find the lowest comparison count available
    min_comps = min(img["comparison_count"] for img in all_images)
    
    # 3. Collect all images with this minimum count (The "Seed Pool")
    seed_pool = [img for img in all_images if img["comparison_count"] == min_comps]
    if len(seed_pool) < 5:
        next_min = min_comps + 1
        seed_pool.extend([img for img in all_images if img["comparison_count"] == next_min])


    # 1. Prepare score-sorted list for binary search
    #all_sorted = sorted(all_images, key=lambda x: x["score"])
    all_sorted = sorted(seed_pool, key=lambda x: x["score"])
    scores_only = [img["score"] for img in all_sorted]

    # 4. Pick a random Seed A
    random.shuffle(seed_pool)
    
    # Load comparison history for checking indirects
    all_comps = get_all_comparisons()
    graph = defaultdict(list)
    compared_pairs = set()
    for c in all_comps:
        A_name, B_name = c["filename_a"], c["filename_b"]
        compared_pairs.add(tuple(sorted((A_name, B_name))))
        if c.get("transitive_depth", 0) > 0: continue
        w = c["winner"]
        l = B_name if w == A_name else A_name
        graph[w].append((l, c.get("weight", 1.0)))

    # Transitive check logic
    
    decay_factor = float(config.get("ranking").get("indirect_impact_decay"))
    

   
    # 5. Find the best Partner B for Seed A using binary search for speed
    for img_a in seed_pool:
        A = img_a["filename"]
        score_a = img_a["score"]
        
        # Calculate allowed range
        allowed_diff = max(0.05, 1.0 * math.exp(-0.3 * img_a["comparison_count"]))
        
        target_low = score_a - allowed_diff
        target_high = score_a + allowed_diff
        
        # Binary search for boundaries
        idx_low = bisect.bisect_left(scores_only, target_low)
        idx_high = bisect.bisect_right(scores_only, target_high) - 1
        
        # We test the boundaries (extreme low and extreme high) to find the max distance
        test_indices = [idx_low, idx_low + 1, idx_high, idx_high - 1]
        test_indices = sorted(list(set(i for i in test_indices if 0 <= i < len(all_sorted))))
        


        # Sort by distance from A descending
        test_indices.sort(key=lambda i: abs(all_sorted[i]["score"] - score_a), reverse=True)
        


        for idx in test_indices:
            img_b = all_sorted[idx]
            B = img_b["filename"]
            if A == B: continue
            if tuple(sorted((A, B))) in compared_pairs: continue
            
            diff = abs(score_a - img_b["score"])
            if diff <= allowed_diff:
                # Check indirect before returning
                winner, impact, depth = check_indirect(A, B,graph)
                if winner:
                    record_comparison(A, B, winner, impact_factor=impact * decay_factor, transitive_depth=depth)
                    compared_pairs.add(tuple(sorted((A, B))))
                    continue
                return (A, B)

    # Absolute fallback
    if len(all_images) >= 2:
        return (all_images[0]["filename"], all_images[1]["filename"])
    return None


def update_scores_after_comparison(
    winner_filename: str,
    loser_filename: str,
    winner_data: dict,
    loser_data: dict,
    impact_factor: float = 1.0,
) -> Tuple[dict, dict]:
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

    comp_id = add_comparison(filename_a, filename_b, winner, impact_factor, transitive_depth)
    if not comp_id: return False

    update_image_score(winner_filename, winner_data["score"])
    update_image_confidence(winner_filename, calculate_confidence(winner_filename, winner_data["score"], winner_data["comparison_count"]), winner_data["comparison_count"])
    update_image_score(loser_filename, loser_data["score"])
    update_image_confidence(loser_filename, calculate_confidence(loser_filename, loser_data["score"], loser_data["comparison_count"]), loser_data["comparison_count"])

    ts = datetime.utcnow().isoformat()
    entry_winner = {"comparison_id": comp_id, "other": loser_filename, "winner": True, "weight": impact_factor, "transitive_depth": transitive_depth, "timestamp": ts}
    entry_loser = {"comparison_id": comp_id, "other": winner_filename, "winner": False, "weight": impact_factor, "transitive_depth": transitive_depth, "timestamp": ts}

    try:
        append_comparison_history_to_json(winner_filename, entry_winner, new_score=winner_data["score"], new_confidence=calculate_confidence(winner_filename, winner_data["score"], winner_data["comparison_count"]))
        append_comparison_history_to_json(loser_filename, entry_loser, new_score=loser_data["score"], new_confidence=calculate_confidence(loser_filename, loser_data["score"], loser_data["comparison_count"]))
    except Exception: pass

    return True
