import os
from pathlib import Path
from typing import Any
from flask import send_from_directory, abort
from urllib.parse import unquote

from numpy import size
from shared.io import load_single_entry_mapping, discover_files, collect_valid_files
from .cache import add, get_all, set_absolute_total, get_cached_metadata
from shared.config import PROJECT_ROOT, config
import random
import matplotlib.pyplot as plt

IMAGE_EXTENSIONS: set[str] = {".png", ".jpg", ".jpeg", ".webp"}
_image_root = None
_image_list_cache: list[str] = []


def serve_file(subpath: str) -> Any:
    """Serve file relative to image_root, properly decoded."""
    root = Path(image_root())
    path: Path = root / unquote(subpath)
    if not path.exists() or not path.is_file():
        abort(404)
    return send_from_directory(str(path.parent), path.name, conditional=True)


def get_json_path(img_path: str) -> str:
    return os.path.splitext(img_path)[0] + ".json"


def load_metadata(image_path: str) -> dict[str, Any] | None:
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


def scan_batch(root: str, limit: int = 100) -> bool:
    """
    Scan images and add to database with their metadata.
    Collects both scored AND unscored files, organizing by score.

    Returns:
        True if any new unscored files were found
    """
    global _image_list_cache
    if len(_image_list_cache) == 0:
        _image_list_cache = get_all(False)

    # Get all image-json file pairs from root
    all_file_pairs = list(discover_files(root))
    if not all_file_pairs:
        return False
    collected_valid_files = collect_valid_files(
        all_file_pairs,
        set(_image_list_cache),
        root,
        max_workers=40,
        scored_only=False,
        limit=1000,
    )
    if len(collected_valid_files) == 0:
        absolute_total = len(_image_list_cache)
    else:
        absolute_total = int(max(len(all_file_pairs), len(all_file_pairs) / 2))

    set_absolute_total(absolute_total)

    if len(collected_valid_files) == 0:
        return False

    random.shuffle(collected_valid_files)  # Shuffle to ensure random processing order

    added_any_unscored = False
    processed_count = 0

    for img_path, entry, _, file_id in collected_valid_files:
        _image_list_cache.append(img_path)
        # Get score if available
        score = entry.get("score")
        if score is not None:
            if score < 1:
                print(f"file {file_id} has a negative score")
                score = 2
            elif score > 5:
                score = 4

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


def get_unscored_images(root: str) -> list[str]:
    """
    Return **truly unscored images** for the client.
    Only images with score=NULL in database.
    Paths are relative to image_root and use forward slashes.
    """
    # global _image_list_cache

    unscored_paths = get_all(unscored_only=True)
    root_path = Path(root)
    unscored: list[str] = []

    for img in unscored_paths:
        try:
            rel_path = str(Path(img).relative_to(root_path))
        except ValueError:
            rel_path = os.path.basename(img)
        unscored.append(rel_path)

    return unscored


def print_slot_distribution(buckets, precision=0, limit=11):
    print(f"\nBucket distribution with precision {precision}:")

    scale = 1.1**precision
    # Base font sizes to be scaled
    base_fs = 12

    # Scale figure size
    plt.figure(figsize=(10 * scale, 6 * scale))

    bucket_keys = sorted(buckets.keys())
    counts = [len(buckets[key]) for key in bucket_keys]

    # width=0.8/scale is already good for the bars
    plt.bar(bucket_keys, counts, width=0.1 * (1 / scale))

    # Scale Labels and Title
    plt.xlabel("Score Buckets", fontsize=base_fs * scale)
    plt.ylabel("Number of Images", fontsize=base_fs * scale)
    plt.title(
        f"Image Distribution (Precision: {precision})", fontsize=(base_fs + 4) * scale
    )

    # Scale Tick Labels (the numbers on the axes)
    plt.xticks(bucket_keys, fontsize=base_fs * scale)
    plt.yticks(fontsize=base_fs * scale)

    # Scale Grid thickness
    plt.grid(axis="y", linestyle="--", alpha=0.7, linewidth=1 * scale)

    # Adjust layout to prevent clipping
    plt.tight_layout()
    plt.show()


ImageSlot = list[dict[str, str | float | int | None]]
ImageBucket = dict[float, ImageSlot]


def move_to_slot(
    image_buckets: ImageBucket,
    current_slot: float,
    target_slot: float,
    target_size: int,
) -> ImageBucket:
    source_list: ImageSlot = image_buckets[current_slot]
    num_to_move: int = len(source_list) - target_size

    image_buckets[target_slot].sort(key=lambda x: int(x["comparison_count"]))

    if num_to_move > 0:
        image_buckets[target_slot].extend(source_list[:num_to_move])
        image_buckets[current_slot] = source_list[num_to_move:]

    return image_buckets
