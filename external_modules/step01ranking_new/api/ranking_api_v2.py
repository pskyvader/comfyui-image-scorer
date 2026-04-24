"""Ranking API v2 - new endpoints for ranking system."""

import sys
import json
from pathlib import Path
import math
# Set up path for shared imports BEFORE any other imports
_root = Path(__file__).parent.parent.parent  # comfyui-image-scorer
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from flask import Blueprint, request, jsonify, current_app
from database.images_table import (
    get_all_images,
)
from database.images_table import get_image_count
from shared.config import config
from shared.paths import image_root
from database.comparisons_table import get_total_comparisons
from algorithm.merge_sort_ranker import select_pair_for_comparison, record_comparison


ranking_bp = Blueprint("ranking_v2", __name__, url_prefix="/api/v2/ranking")


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
    print("[API] get_status() called")
    try:
        # Trigger background scan on status check (e.g. on page reload)
        try:
            if hasattr(current_app, "image_processor"):
                import threading
                threading.Thread(
                    target=current_app.image_processor.process_next_batch,
                    args=(image_root,),
                    daemon=True
                ).start()
        except Exception: pass

        all_images = get_all_images()
        total = len(all_images)
        ranked = len([img for img in all_images if img["comparison_count"] > 0])
        unranked = total - ranked

        total_comps = get_total_comparisons()

        avg_conf = (
            sum(img["confidence"] for img in all_images) / total if total > 0 else 0.0
        )

        # Ensure min_images is available for the frontend
        try:
            min_images = int(config["ranking"]["lru_size"])
        except Exception:
            min_images = 100

        return jsonify(
            {
                "total_images": total,
                "ranked_images": ranked,
                "unranked_images": unranked,
                "total_comparisons": total_comps,
                "average_confidence": round(avg_conf, 3),
                "min_images": min_images,
            }
        )
    except Exception as e:
        print(f"Error getting status: {e}")
        return jsonify({"error": str(e)}), 500


@ranking_bp.route("/next-pair", methods=["GET"])
def get_next_pair():
    """
    Get next pair of images for comparison.
    
    Query params:
    - exclude: JSON array of filenames to exclude from the pair

    Returns JSON with:
    {
        left: {
            filename: str,
            score: float (0-1),
            confidence: float (0-1),
            comparison_count: int
        },
        right: { ... }
    }

    or 204 No Content if no pairs available
    """
    try:
        processor = getattr(current_app, "image_processor", None)
        excluded_files_set = set()
        if processor:
            with processor.recent_lock:
                excluded_files_set = set(processor.recent_images)
        
        # Enforce minimum number of images before comparisons are allowed.
        lru_size = processor.lru_size if processor else int(config["ranking"]["lru_size"])
        total_images = get_image_count()
        
        if total_images < lru_size:
            return jsonify({
                "error": "Not Enough Images",
                "message": f"The system requires at least {lru_size} valid images to start the ranking process, but only {total_images} were found in the database."
            }), 400

        # Try to get a pair that doesn't include excluded images
        max_retries = 10
        for attempt in range(max_retries):
            pair = select_pair_for_comparison(exclude_set=excluded_files_set)
            if not pair: return "", 204

            filename_a, filename_b = pair

            # Skip if either image is in the excluded set
            if filename_a in excluded_files_set or filename_b in excluded_files_set:
                continue

            # Get full data for both images
            from database.images_table import get_image as get_img_data
            data_a = get_img_data(filename_a)
            data_b = get_img_data(filename_b)

            if not data_a or not data_b or data_a["filename"] == data_b["filename"]:
                continue

            # Track these images as "recently shown" on server side
            if processor:
                with processor.recent_lock:
                    processor.recent_images.append(filename_a)
                    processor.recent_images.append(filename_b)


            return jsonify(
                {
                    "left": {
                        "filename": data_a["filename"],
                        "score": round(data_a["score"], 3),
                        "confidence": round(data_a["confidence"], 3),
                        "comparison_count": data_a["comparison_count"],
                    },
                    "right": {
                        "filename": data_b["filename"],
                        "score": round(data_b["score"], 3),
                        "confidence": round(data_b["confidence"], 3),
                        "comparison_count": data_b["comparison_count"],
                    },
                    "rationale": {
                        "seed": data_a["filename"],
                        "partner": data_b["filename"],
                        "score_diff": round(abs(data_a["score"] - data_b["score"]), 3),
                        "allowed_range": round(max(0.05, 1.0 * math.exp(-0.3 * data_a["comparison_count"])), 3),
                        "strategy": "Random Seed + Max Distance"
                    }
                }
            )
        
        # If we exhausted retries, return no content
        return "", 204
    except Exception as e:
        print(f"Error getting next pair: {e}")
        return jsonify({"error": str(e)}), 500


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
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Missing request body"}), 400

        filename_a = data.get("filename_a")
        filename_b = data.get("filename_b")
        winner = data.get("winner")

        if not all([filename_a, filename_b, winner]):
            return jsonify({"error": "Missing required fields"}), 400

        if filename_a == filename_b:
            return jsonify({"error": "Cannot compare image to itself"}), 400

        if winner not in [filename_a, filename_b]:
            return jsonify({"error": "Winner must be one of the images"}), 400

        # Record comparison
        success = record_comparison(filename_a, filename_b, winner)

        if not success:
            return jsonify({"error": "Failed to record comparison"}), 500

        # Get updated data
        from database.images_table import get_image as get_img_data

        data_a = get_img_data(filename_a)
        data_b = get_img_data(filename_b)

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
    except Exception as e:
        print(f"Error submitting comparison: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        # Trigger background scan for next batch
        try:
            if hasattr(current_app, "image_processor"):
                import threading
                threading.Thread(
                    target=current_app.image_processor.process_next_batch,
                    args=(image_root,),
                    daemon=True
                ).start()
        except Exception as e:
            print(f"Failed to trigger background scan: {e}")



def register_ranking_routes(app) -> None:
    """Register all ranking API v2 routes with Flask app."""
    app.register_blueprint(ranking_bp)
@ranking_bp.route("/sync-all", methods=["POST"])
def sync_all_to_json():
    """Sync all current DB scores/confidence to their JSON files (Backup)."""
    try:
        from database.images_table import get_all_images
        from file_management.path_handler import sync_image_metadata_to_json
        
        images = get_all_images()
        count = 0
        errors = 0
        
        for img in images:
            success = sync_image_metadata_to_json(
                img["filename"], 
                img["score"], 
                img["confidence"], 
                img["comparison_count"]
            )
            if success: count += 1
            else: errors += 1
            
        return jsonify({
            "status": "success",
            "synced_count": count,
            "error_count": errors
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
