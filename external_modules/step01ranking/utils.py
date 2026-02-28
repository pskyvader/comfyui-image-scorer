import os
from pathlib import Path
from typing import Any, List, Dict
from flask import send_from_directory, abort
from urllib.parse import unquote
from shared.io import load_single_entry_mapping, discover_files, collect_valid_files
from .cache import add, get, set_absolute_total, get_cached_metadata
from shared.config import PROJECT_ROOT, config
import random

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
_image_root = None
_image_list_cache: List[str] = []


def serve_file(subpath: str) -> Any:
    """Serve file relative to image_root, properly decoded."""
    root = Path(image_root())
    path = root / unquote(subpath)
    if not path.exists() or not path.is_file():
        abort(404)
    return send_from_directory(str(path.parent), path.name, conditional=True)


def get_json_path(img_path: str) -> str:
    return os.path.splitext(img_path)[0] + ".json"


def load_metadata(image_path: str) -> Dict[str, Any] | None:
    metadata_path = get_json_path(image_path)
    payload, ts, err = load_single_entry_mapping(metadata_path)
    if err:
        return None
    return {ts: payload}


def image_root() -> str:
    global _image_root
    if _image_root:
        return _image_root
    img_root = config["image_root"]
    root_path = Path(img_root)
    if not root_path.is_absolute():
        root_path = PROJECT_ROOT.joinpath(root_path).resolve()
    _image_root = str(root_path)
    return _image_root


# def find_images(root_dir: str) -> List[str]:
#     files: List[str] = []
#     for path, _, filenames in os.walk(root_dir):
#         for name in filenames:
#             if Path(name).suffix.lower() in IMAGE_EXTENSIONS:
#                 # files.append(os.path.join(path, name).replace("\\", "/"))
#                 full_path = os.path.join(path, name)
#                 files.append(full_path.replace("\\", "/"))

#     random.shuffle(files)
#     set_absolute_total(len(files))
#     return files


def scan_batch(root: str, limit: int = 100) -> bool:
    """
    Scan images and add to database with their metadata.
    Collects both scored AND unscored files, organizing by score.

    Returns:
        True if any new unscored files were found
    """
    global _image_list_cache

    # Get all image-json file pairs from root
    all_file_pairs = list(discover_files(root))
    if not all_file_pairs:
        return False
    collected_valid_files = collect_valid_files(
        all_file_pairs, set(_image_list_cache), root, limit=limit, scored_only=False
    )
    set_absolute_total(len(_image_list_cache) + len(collected_valid_files))

    random.shuffle(collected_valid_files)  # Shuffle to ensure random processing order

    added_any_unscored = False
    processed_count = 0

    for img_path, entry, _, file_id in collected_valid_files:
        _image_list_cache.append(file_id)
        # Get score if available
        score = entry.get("score")
        comparison_count = entry.get("comparison_count", 0)
        score_modifier = entry.get("score_modifier", 0)

        from_bd = get_cached_metadata(img_path)
        if from_bd and from_bd["score"] is not None:
            if score == from_bd["score"]:
                continue

        # Add to database with metadata
        add(
            img_path,
            score=score,
            comparison_count=comparison_count,
            score_modifier=score_modifier,
        )

        # Track unscored files
        if score is None:
            added_any_unscored = True

        processed_count += 1

        if limit > 0 and processed_count >= limit:
            break

    return added_any_unscored


def get_unscored_images(root: str) -> List[str]:
    """
    Return **truly unscored images** for the client.
    Only images with score=NULL in database.
    Paths are relative to image_root and use forward slashes.
    """
    # global _image_list_cache

    unscored_paths = get(unscored_only=True)
    root_path = Path(root)
    unscored: List[str] = []

    for img in unscored_paths:
        try:
            rel_path = str(Path(img).relative_to(root_path))
        except ValueError:
            rel_path = os.path.basename(img)
        unscored.append(rel_path)

    return unscored
