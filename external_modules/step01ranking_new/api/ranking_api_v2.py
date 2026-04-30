"""Ranking API v2 - new endpoints for ranking system."""

import sys
import math
import logging
from typing import Any
from pathlib import Path
from collections import deque

_root = Path(__file__).parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from flask import Blueprint, request, jsonify, current_app
from database.images_table import get_all_images, get_image_count, get_image as get_img_data
from shared.config import config
from database.comparisons_table import get_total_comparisons, get_skipped_comparison_count
from algorithm.merge_sort_ranker import select_pair_for_comparison, record_comparison


ranking_bp = Blueprint("ranking_v2", __name__, url_prefix="/api/v2/ranking")
logger = logging.getLogger(__name__)


def _get_processor() -> Any:
    """Get the image processor from the current Flask app."""
    return getattr(current_app, "image_processor", None) or current_app.extensions.get("image_processor")


@ranking_bp.route("/status", methods=["GET"])
def get_status():
    """
    Get ranking system status.

    Returns JSON with:
    - total_images: Total images in system
    - ranked_images: Images with scores
    - unranked_images: Images without scores
    - total_comparisons: Total comparisons made
    - average_confidence: Average confidence across images
    """
    all_images = get_all_images()
    total = len(all_images)
    ranked = len([img for img in all_images if img["comparison_count"] > 0])
    unranked = total - ranked

    total_comps = get_total_comparisons()

    avg_conf = (
        sum(img["confidence"] for img in all_images) / total if total > 0 else 0.0
    )

    min_images = int(config["ranking"]["lru_size"])

    skipped_count = get_skipped_comparison_count()

    return jsonify(
        {
            "total_images": total,
            "ranked_images": ranked,
            "unranked_images": unranked,
            "total_comparisons": total_comps,
            "skipped_comparisons": skipped_count,
            "average_confidence": round(avg_conf, 3),
            "min_images": min_images,
        }
    )


@ranking_bp.route("/next-pair", methods=["GET"])
def get_next_pair():
    """
    Get next pair of images for comparison.
    """
    processor = _get_processor()
    excluded_files_set: set[str] = set()

    if processor:
        with processor.recent_lock:
            excluded_files_set = set(processor.recent_images)

    lru_size = int(config["ranking"]["lru_size"])
    total_images = get_image_count()

    if total_images < lru_size:
        return (
            jsonify(
                {
                    "error": "Not Enough Images",
                    "message": f"The system requires at least {lru_size} valid images to start the ranking process, but only {total_images} were found in the database.",
                }
            ),
            400,
        )

    max_retries = 20
    logger.debug(f"[NEXT-PAIR] Excluded set: {len(excluded_files_set)} images")

    for attempt in range(max_retries):
        pair = select_pair_for_comparison(exclude_set=excluded_files_set)
        if not pair:
            logger.debug(f"[NEXT-PAIR] No pair found after {attempt+1} attempts")
            return "", 204

        filename_a, filename_b = pair

        if filename_a in excluded_files_set or filename_b in excluded_files_set:
            logger.debug(f"[NEXT-PAIR] Retry: {filename_a} or {filename_b} in excluded set")
            continue

        if filename_a == filename_b:
            logger.debug(f"[NEXT-PAIR] Retry: same image {filename_a}")
            continue

        data_a = get_img_data(filename_a)
        data_b = get_img_data(filename_b)

        if not data_a or not data_b or data_a["filename"] == data_b["filename"]:
            continue

        if processor:
            with processor.recent_lock:
                processor.recent_images.append(filename_a)
                processor.recent_images.append(filename_b)

        return jsonify(
            {
                "left": {
                    "filename": data_a["filename"],
                    "score": round(data_a["score"], 4),
                    "confidence": round(data_a["confidence"], 4),
                    "comparison_count": data_a["comparison_count"],
                },
                "right": {
                    "filename": data_b["filename"],
                    "score": round(data_b["score"], 4),
                    "confidence": round(data_b["confidence"], 4),
                    "comparison_count": data_b["comparison_count"],
                },
                "rationale": {
                    "seed": data_a["filename"],
                    "partner": data_b["filename"],
                    "score_diff": round(abs(data_a["score"] - data_b["score"]), 4),
                    "common_confidence": round(max(data_a["confidence"], data_b["confidence"]), 4),
                    "allowed_range": round(max(0.05, 1.0 * math.exp(-0.3 * data_a["comparison_count"])), 4),
                    "strategy": "Lowest Common Confidence + Chain Ends + Score Gap Cap",
                },
            }
        )

    return "", 204


@ranking_bp.route("/submit-comparison", methods=["POST"])
def submit_comparison():
    """
    Submit comparison result.

    Expected JSON:
    {
        filename_a: str,
        filename_b: str,
        winner: str (must be one of the filenames),
    }

    Returns JSON with updated scores for both images.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing request body"}), 400

    filename_a = data.get("filename_a")
    filename_b = data.get("filename_b")
    winner = data.get("winner")

    logger.info(f"[SUBMIT] Received: {filename_a} vs {filename_b}, winner: {winner}")

    if not all([filename_a, filename_b, winner]):
        return jsonify({"error": "Missing required fields"}), 400

    if filename_a == filename_b:
        return jsonify({"error": "Cannot compare image to itself"}), 400

    if winner not in [filename_a, filename_b]:
        return jsonify({"error": "Winner must be one of the images"}), 400

    success = record_comparison(filename_a, filename_b, winner)

    if not success:
        return jsonify({"error": "Failed to record comparison"}), 500

    # Submitted images should remain in the LRU cache to prevent immediate re-selection.
    # The cache automatically manages removal of oldest items when the limit is reached.


    data_a = get_img_data(filename_a)
    data_b = get_img_data(filename_b)

    if data_a is None or data_b is None:
        return jsonify({"error": "Image not found"}), 404

    return jsonify(
        {
            "ok": True,
            "images": {
                filename_a: {
                    "score": round(data_a["score"], 3),
                    "confidence": round(data_a["confidence"], 3),
                    "comparison_count": data_a["comparison_count"],
                },
                filename_b: {
                    "score": round(data_b["score"], 3),
                    "confidence": round(data_b["confidence"], 3),
                    "comparison_count": data_b["comparison_count"],
                },
            },
        }
    )


def register_ranking_routes(app) -> None:
    """Register all ranking API v2 routes with Flask app."""
    app.register_blueprint(ranking_bp)


@ranking_bp.route("/sync-all", methods=["POST"])
def sync_all_to_json():
    """Sync all current DB scores/confidence to their JSON files (Backup)."""
    try:
        from file_management.path_handler import sync_image_metadata_to_json

        images = get_all_images()
        count = 0
        errors = 0

        for img in images:
            success = sync_image_metadata_to_json(
                img["filename"],
                img["score"],
                img["confidence"],
                img["comparison_count"],
            )
            if success:
                count += 1
            else:
                errors += 1

        return jsonify(
            {"status": "success", "synced_count": count, "error_count": errors}
        )
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
