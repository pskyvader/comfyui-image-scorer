"""Path handler - compute tier structure from scores (filename-only DB logic)."""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Any
import math
import time
import shutil

logger = logging.getLogger(__name__)

# Import from parent shared folder
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from shared.config import config
from shared.paths import image_root_processed


def find_workspace_root() -> Path:
    # Walk ancestors and pick the first directory that looks like the workspace root.
    # Heuristic: workspace root contains a top-level `custom_nodes` folder or `main.py`.
    p = Path(__file__).resolve()
    for ancestor in p.parents:
        if (ancestor / "main.py").exists():
            return ancestor
        if (ancestor / "custom_nodes").is_dir():
            return ancestor
    # Fallback to the highest ancestor available
    return Path(__file__).resolve().parents[-1]


def get_ranked_root() -> Path:
    """Get root folder for all ranked images."""
    root_path = Path(image_root_processed)
    if not root_path.is_absolute():
        workspace_root = find_workspace_root()
        root_path = workspace_root / root_path
    root_path.mkdir(parents=True, exist_ok=True)
    return root_path


def compute_path_from_filename(filename: str, score: float) -> Path:
    """Compute full folder path for image based on score."""

    ranked_root = get_ranked_root()
    score_truncated = math.floor(score * 10) / 10.0
    base_folder = ranked_root / f"scored_{score_truncated:.1f}"
    threshold = config["ranking"]["subfolder_threshold"]

    has_subfolders = False
    file_count = 0
    if base_folder.exists():
        try:
            items = os.listdir(base_folder)
            file_count = len(items)
            for item in items:
                if item.startswith("scored_") and (base_folder / item).is_dir():
                    has_subfolders = True
                    break
        except Exception:
            file_count = 0

    if (file_count < threshold and not has_subfolders) or score_truncated >= 1.0:
        return base_folder / filename

    score_second = math.floor(score * 100) / 100.0
    second_folder = base_folder / f"scored_{score_second:.2f}"
    return second_folder / filename


def find_image_path(filename: str) -> Path | None:
    """Find current path for a filename inside ranked root (exact name search)."""
    ranked_root = get_ranked_root()
    try:
        for root, dirs, files in os.walk(ranked_root):
            if filename in files:
                return Path(root) / filename
    except Exception:
        pass
    return None


def append_comparison_history_to_json(
    filename: str,
    comparison_data: dict[str, Any],
    new_score: float | None = None,
    new_confidence: float | None = None,
) -> bool:
    """Sync full comparison history from DB to JSON metadata.
    Since DB is the source of truth, we rebuild history entirely from DB.
    This ensures deleted comparisons are permanently removed from JSON as well.
    """

    img_path = find_image_path(filename)
    if not img_path:
        raise FileNotFoundError(
            f"Unable to locate ranked image for JSON history update: {filename}"
        )

    json_path = img_path.with_suffix(".json")
    if not json_path.exists():
        raise FileNotFoundError(
            f"JSON metadata file not found for image {filename} at {json_path}"
        )

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.exception(f"Failed to read JSON metadata for {filename}: {e}")
        return False

    # Rebuild comparison history entirely from database
    # This ensures deleted comparisons stay deleted
    from database.comparisons_table import get_all_comparisons
    from database.images_table import get_image as get_image_data
    all_comparisons = get_all_comparisons()
    history = []
    for comp in all_comparisons:
        if comp["filename_a"] == filename or comp["filename_b"] == filename:
            is_winner = comp["winner"] == filename
            other = comp["filename_b"] if comp["filename_a"] == filename else comp["filename_a"]
            other_data = get_image_data(other)
            history.append({
                "comparison_id": comp["id"],
                "other": other,
                "opponent_score": other_data["score"] if other_data else 0.5,
                "winner": is_winner,
                "weight": float(comp["weight"]) if comp["weight"] is not None else 1.0,
                "transitive_depth": int(comp["transitive_depth"]) if comp["transitive_depth"] is not None else 0,
                "timestamp": comp["timestamp"],
            })

    data["comparison_history"] = history
    data["comparison_count"] = len(history)
    logger.info(f"[HISTORY SYNC] Rebuilt {len(history)} comparison history entries for {filename} from database")

    if new_score is not None:
        data["score"] = new_score
    if new_confidence is not None:
        data["confidence"] = new_confidence

    # Keep confidence in sync
    target_comparisons = 5
    try:
        target_comparisons = int(
            config["ranking"]["target_comparisons_for_max_confidence"]
        )
    except Exception:
        target_comparisons = 5

    try:
        history_sorted = sorted(
            history, key=lambda x: x.get("timestamp", ""), reverse=True
        )
        total_weight = sum(
            float(c.get("weight", 1.0)) * (0.5 ** (i / 9.0))
            for i, c in enumerate(history_sorted)
        )
        target_divisor = sum(0.5 ** (i / 9.0) for i in range(target_comparisons))
        data["confidence"] = min(1.0, total_weight / target_divisor)
    except Exception as e:
        logger.exception(f"Failed to recalculate confidence for {filename}: {e}")

    # Write JSON first
    try:
        tmp = json_path.parent / (json_path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(str(tmp), str(json_path))
    except Exception as e:
        raise IOError(f"Failed to write JSON metadata for {filename}: {e}")

    # Move files if score changed - with retry logic for locked files
    if new_score is not None:
        target_path = compute_path_from_filename(filename, new_score)
        if target_path.parent != img_path.parent:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Retry move up to 5 times with backoff
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    os.replace(str(img_path), str(target_path))
                    os.replace(str(json_path), target_path.with_suffix(".json"))
                    break
                except PermissionError as e:
                    if attempt < max_retries - 1:
                        wait_time = 0.5 * (2 ** attempt)  # 0.5s, 1s, 2s, 4s
                        logger.debug(f"File locked, retrying in {wait_time}s: {filename}")
                        time.sleep(wait_time)
                    else:
                        # Move failed but JSON was updated - log and continue
                        logger.warning(f"Could not move file after {max_retries} attempts: {filename}")
                        return True
            # Update path reference for future operations
            img_path = target_path

    return True


def sync_image_metadata_to_json(
    filename: str, score: float, confidence: float, comparison_count: int, all_comparisons: list | None = None
) -> bool:
    """Updates JSON metadata from DB including score, confidence, comparison_count, and full comparison history.
    
    Args:
        all_comparisons: Optional pre-fetched list of all comparisons. If not provided, will fetch from DB.
    """
    img_path = find_image_path(filename)
    if not img_path:
        return False
    json_path = img_path.with_suffix(".json")
    if not json_path.exists():
        return False

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Update basic fields
        data["score"] = score
        data["confidence"] = confidence
        data["comparison_count"] = comparison_count

        # Rebuild comparison history from database
        if all_comparisons is None:
            from database.comparisons_table import get_all_comparisons
            all_comparisons = get_all_comparisons()
        
        from database.images_table import get_image as get_image_data
        history = []
        for comp in all_comparisons:
            # Check if this image is involved in the comparison
            if comp["filename_a"] == filename or comp["filename_b"] == filename:
                # Determine if this image won
                is_winner = comp["winner"] == filename
                # The "other" image is the one that isn't this filename
                other = comp["filename_b"] if comp["filename_a"] == filename else comp["filename_a"]
                other_data = get_image_data(other)
                history.append({
                    "comparison_id": comp["id"],
                    "other": other,
                    "opponent_score": other_data["score"] if other_data else 0.5,
                    "winner": is_winner,
                    "weight": float(comp["weight"]) if comp["weight"] is not None else 1.0,
                    "transitive_depth": int(comp["transitive_depth"]) if comp["transitive_depth"] is not None else 0,
                    "timestamp": comp["timestamp"],
                })

        data["comparison_history"] = history
        data["comparison_count"] = len(history)

        # Atomic write
        tmp = json_path.parent / (json_path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(str(tmp), str(json_path))

        # Skip file moving - just update JSON. The server will find the file via recursive search.
        return True
    except Exception:
        return False
