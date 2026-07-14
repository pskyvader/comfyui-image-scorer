"""Path handler - compute tier structure from scores and sync companion JSON."""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any

from ...shared.logger import get_logger, ModuleLogger

logger: ModuleLogger = get_logger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from ...shared.config import config
from ...shared.io import atomic_write_json, load_json
from ...shared.paths import image_root_processed
from .comparisons_table import (
    get_all_comparisons,
)
from .images_table import get_image as get_image_data
import time

_folder_listdir_cache: dict[str, tuple[int, bool]] = {}


def prewarm_folder_cache(ranked_root: Path) -> None:
    """Eagerly populate tier folder cache for all scored_X.X directories."""
    for i in range(11):
        base = ranked_root / f"scored_{i / 10:.1f}"
        if not base.exists():
            continue
        try:
            items = os.listdir(base)
            has_subfolders = any(
                item.startswith("scored_") and (base / item).is_dir() for item in items
            )
            _folder_listdir_cache[str(base)] = (len(items), has_subfolders)
        except Exception:
            pass


def clear_folder_cache() -> None:
    _folder_listdir_cache.clear()


def find_workspace_root() -> Path:
    _start = time.perf_counter()
    _start = time.perf_counter()
    p = Path(__file__).resolve()
    for ancestor in p.parents:
        if (ancestor / "main.py").exists():
            result = ancestor

            result = result

            return result
        if (ancestor / "custom_nodes").is_dir():
            result = ancestor

            result = result

            return result
    result = Path(__file__).resolve().parents[-1]
    return result


def get_ranked_root() -> Path:
    _start = time.perf_counter()
    _start = time.perf_counter()
    root_path = Path(image_root_processed)
    if not root_path.is_absolute():
        root_path = find_workspace_root() / root_path
    root_path.mkdir(parents=True, exist_ok=True)
    result = root_path
    return result


def compute_path_from_filename(filename: str, score: float) -> Path:
    _start = time.perf_counter()
    ranked_root = get_ranked_root()
    clamped_score = max(0.0, min(1.0, float(score)))
    score_truncated = math.floor(clamped_score * 10) / 10.0
    base_folder = ranked_root / f"scored_{score_truncated:.1f}"
    threshold = int(config["ranking"]["subfolder_threshold"])

    cache_key = str(base_folder)
    cached = _folder_listdir_cache.get(cache_key)
    if cached is not None:
        file_count, has_subfolders = cached
    elif base_folder.exists():
        try:
            items = os.listdir(base_folder)
            file_count = len(items)
            has_subfolders = any(
                item.startswith("scored_") and (base_folder / item).is_dir()
                for item in items
            )
            _folder_listdir_cache[cache_key] = (file_count, has_subfolders)
        except Exception:
            file_count = 0
            has_subfolders = False
    else:
        file_count = 0
        has_subfolders = False

    if (file_count < threshold and not has_subfolders) or score_truncated >= 1.0:
        result = base_folder / filename
        return result

    score_second = math.floor(clamped_score * 100) / 100.0
    result = base_folder / f"scored_{score_second:.2f}" / filename
    return result


def find_image_path(filename: str) -> Path | None:
    ranked_root = get_ranked_root()
    try:
        for root, _, files in os.walk(ranked_root):
            if filename in files:
                result = Path(root) / filename
                return result
    except Exception:
        pass
    result = None
    return result


def _build_history_for_filename(
    filename: str,
    all_comparisons: list[dict[str, Any]] | None = None,
    filename_to_comparisons: dict[str, list[dict[str, Any]]] | None = None,
    filename_to_image_data: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if filename_to_comparisons is not None:
        comps = filename_to_comparisons[filename]
    elif all_comparisons is not None:
        comps = [
            c
            for c in all_comparisons
            if c["filename_a"] == filename or c["filename_b"] == filename
        ]
    else:
        comps = []

    history: list[dict[str, Any]] = []
    for comp in comps:
        is_winner = comp["winner"] == filename
        other = (
            comp["filename_b"] if comp["filename_a"] == filename else comp["filename_a"]
        )
        if filename_to_image_data is not None:
            other_data = filename_to_image_data[other]
        else:
            other_data = get_image_data(other)
        history.append(
            {
                "comparison_id": comp["id"],
                "other": other,
                "opponent_score": other_data["score"] if other_data else 0.5,
                "winner": is_winner,
                "weight": float(comp["weight"]) if comp["weight"] is not None else 1.0,
                "transitive_depth": (
                    int(comp["transitive_depth"])
                    if comp["transitive_depth"] is not None
                    else 0
                ),
                "timestamp": comp["timestamp"],
            }
        )
    history.sort(key=lambda item: (item["timestamp"], item["comparison_id"]))
    return history


def _move_image_and_json(current_image: Path, current_json: Path, score: float) -> None:
    _start = time.perf_counter()
    target_path = compute_path_from_filename(current_image.name, score)
    if target_path.parent == current_image.parent:
        return
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_json = target_path.with_suffix(".json")
    os.replace(str(current_image), str(target_path))
    os.replace(str(current_json), str(target_json))


def sync_image_metadata_to_json(
    filename: str,
    score: float,
    rating_mu: float,
    rating_sigma: float,
    comparison_count: int,
    all_comparisons: list[dict[str, Any]] | None = None,
    filename_to_path: dict[str, Path] | None = None,
    filename_to_comparisons: dict[str, list[dict[str, Any]]] | None = None,
    filename_to_image_data: dict[str, dict[str, Any]] | None = None,
    filename_to_entry: dict[str, dict[str, Any]] | None = None,
) -> bool:
    """Rewrite one JSON companion file from DB-backed state."""

    if filename_to_path is not None and filename in filename_to_path:
        img_path = filename_to_path[filename]
    else:
        img_path = find_image_path(filename)
    if not img_path:
        logger.warning("Image file not found for %s, cannot sync JSON.", filename)
        return False
    json_path = img_path.with_suffix(".json")

    if filename_to_entry is not None:
        data = filename_to_entry.get(filename)
        if data is None:
            return False
    else:
        if not json_path.exists():
            return False
        data, err = load_json(str(json_path), expect=dict, default=None)
        if err is not None or data is None:
            return False

    if filename_to_comparisons is None and all_comparisons is None:
        all_comparisons = get_all_comparisons()

    history = _build_history_for_filename(
        filename,
        all_comparisons=all_comparisons,
        filename_to_comparisons=filename_to_comparisons,
        filename_to_image_data=filename_to_image_data,
    )

    old_score = data.get("score")
    old_mu = data.get("rating_mu")
    old_sigma = data.get("rating_sigma")
    old_count = data.get("comparison_count")
    old_history = data.get("comparison_history") or []

    data["score"] = float(score)
    data["rating_mu"] = float(rating_mu)
    data["rating_sigma"] = float(rating_sigma)
    data["comparison_count"] = int(comparison_count)
    data["comparison_history"] = history
    data.pop("confidence", None)

    if (
        old_score == data["score"]
        and old_mu == data["rating_mu"]
        and old_sigma == data["rating_sigma"]
        and old_count == data["comparison_count"]
        and old_history == data["comparison_history"]
    ):
        return True

    try:
        atomic_write_json(str(json_path), data, indent=2)
        _move_image_and_json(img_path, json_path, score)
        return True
    except Exception as exc:
        logger.error("Failed to sync JSON metadata for %s: %s", filename, exc)
        return False


def append_comparison_history_to_json(
    filename: str,
    comparison_data: dict[str, Any],
    new_score: float | None = None,
    new_rating_mu: float | None = None,
    new_rating_sigma: float | None = None,
) -> bool:
    """Compatibility wrapper that performs a full DB-backed JSON sync."""

    img = get_image_data(filename)
    if img is None:
        return False
    return sync_image_metadata_to_json(
        filename=filename,
        score=float(new_score if new_score is not None else img["score"]),
        rating_mu=float(
            new_rating_mu if new_rating_mu is not None else img["rating_mu"]
        ),
        rating_sigma=float(
            new_rating_sigma if new_rating_sigma is not None else img["rating_sigma"]
        ),
        comparison_count=int(img["comparison_count"]),
        all_comparisons=get_all_comparisons(),
    )
