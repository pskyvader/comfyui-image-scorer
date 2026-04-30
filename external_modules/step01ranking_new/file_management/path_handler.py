"""Path handler - compute tier structure from scores (filename-only DB logic)."""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Any

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
    import math

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
    """Append a comparison result to the image's JSON metadata."""
    import time
    import shutil

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

    history = data.get("comparison_history", [])
    if not isinstance(history, list):
        logger.warning(
            f"Invalid comparison_history format for {filename}; resetting to empty list."
        )
        history = []

    if "timestamp" not in comparison_data:
        comparison_data["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    history.append(comparison_data)
    data["comparison_history"] = history
    data["comparison_count"] = len(history)

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
    filename: str, score: float, confidence: float, comparison_count: int
) -> bool:
    """Updates JSON score, confidence, and comparison_count from DB.
    Does NOT modify history list. Used for full database backups.
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

        data["score"] = score
        data["confidence"] = confidence
        data["comparison_count"] = comparison_count

        # Atomic write
        tmp = json_path.parent / (json_path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(str(tmp), str(json_path))

        # Skip file moving - just update JSON. The server will find the file via recursive search.
        return True
    except Exception:
        return False
