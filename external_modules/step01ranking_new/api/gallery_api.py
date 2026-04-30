"""Gallery API - endpoints for viewing and filtering ranked images."""

import sys
import json
import logging
from pathlib import Path

# Set up path for shared imports BEFORE any other imports
_root = Path(__file__).parent.parent.parent  # comfyui-image-scorer
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from flask import Blueprint, request, jsonify
from file_management.path_handler import find_image_path
from database.images_table import (
    get_all_images,
    get_image,
)


gallery_bp = Blueprint("gallery_v2", __name__, url_prefix="/api/v2/gallery")
logger = logging.getLogger(__name__)


@gallery_bp.route("/images", methods=["GET"])
def list_images():
    """
    List ranked images with optional filtering and pagination.

    Query parameters:
    - page: Page number (1-based)
    - per_page: Items per page
    - score_min: Minimum score (0.0-1.0)
    - score_max: Maximum score (0.0-1.0)
    - confidence_min: Minimum confidence (0.0-1.0)
    - confidence_max: Maximum confidence (0.0-1.0)
    - comparisons_min: Minimum comparison count
    - comparisons_max: Maximum comparison count
    - sort: Sort order (score_desc, score_asc, confidence_desc, confidence_asc, comparisons_desc, comparisons_asc, newest)

    Returns JSON with paginated images and total count.
    """
    try:
        page = request.args.get("page", 1, type=int)
        per_page = request.args.get("per_page", 20, type=int)
        if page < 1:
            page = 1
        if per_page < 1 or per_page > 100:
            per_page = 20

        scored = get_all_images()

        score_min = request.args.get("score_min", 0.0, type=float)
        score_max = request.args.get("score_max", 1.0, type=float)
        confidence_min = request.args.get("confidence_min", 0.0, type=float)
        confidence_max = request.args.get("confidence_max", 1.0, type=float)
        comparisons_min = request.args.get("comparisons_min", 0, type=int)
        comparisons_max = request.args.get("comparisons_max", 999999, type=int)

        filtered = []
        for img in scored:
            if not (score_min <= img["score"] <= score_max):
                continue
            if not (confidence_min <= img["confidence"] <= confidence_max):
                continue
            comp_count = img.get("comparison_count", 0)
            if not (comparisons_min <= comp_count <= comparisons_max):
                continue
            filtered.append(img)

        # Apply sorting
        sort_by = request.args.get("sort", "score_desc")
        if sort_by == "score_asc":
            filtered.sort(key=lambda x: x["score"])
        elif sort_by == "score_desc":
            filtered.sort(key=lambda x: x["score"], reverse=True)
        elif sort_by == "confidence_asc":
            filtered.sort(key=lambda x: x["confidence"])
        elif sort_by == "confidence_desc":
            filtered.sort(key=lambda x: x["confidence"], reverse=True)
        elif sort_by == "comparisons_asc":
            filtered.sort(key=lambda x: x["comparison_count"])
        elif sort_by == "comparisons_desc":
            filtered.sort(key=lambda x: x["comparison_count"], reverse=True)
        elif sort_by == "newest":
            filtered.sort(key=lambda x: x["last_compared_at"] or "", reverse=True)

        # Paginate
        total = len(filtered)
        offset = (page - 1) * per_page
        paginated = filtered[offset : offset + per_page]

        images = [
            {
                "filename": img["filename"],
                "score": round(img["score"], 3),
                "file": f"{img['filename']}?score={round(img['score'],3)}",
                "confidence": round(img["confidence"], 3),
                "comparison_count": img["comparison_count"],
                "tier": int(img["score"] * 10),
            }
            for img in paginated
        ]

        return jsonify(
            {
                "images": images,
                "total": total,
                "page": page,
                "per_page": per_page,
            }
        )
    except Exception as e:
        logger.error(f"Error listing images: {e}")
        return jsonify({"error": str(e)}), 500


@gallery_bp.route("/tier/<int:tier>", methods=["GET"])
def get_tier_images(tier: int):
    """
    Get all images in a specific tier.

    Returns JSON array of images with filenames and scores.
    """
    try:
        if tier < 0 or tier > 9:
            return jsonify({"error": "Tier must be 0-9"}), 400

        all_images = get_all_images()

        result = []
        for img in all_images:
            img_tier = int(img["score"] * 10)
            if img_tier != tier:
                continue

            result.append(
                {
                    "filename": img["filename"],
                    "score": round(img["score"], 3),
                    "file": f"{img['filename']}?score={round(img['score'],3)}",
                    "confidence": round(img["confidence"], 3),
                    "comparison_count": img["comparison_count"],
                }
            )

        return jsonify({"tier": tier, "images": result, "count": len(result)})
    except Exception as e:
        logger.error(f"Error getting tier {tier}: {e}")
        return jsonify({"error": str(e)}), 500


@gallery_bp.route("/search", methods=["GET"])
def search_images():
    """
    Search images by filename pattern or score range.

    Query parameters:
    - query: Filename search pattern
    - score_min, score_max: Score range (0.0-1.0)
    - confidence_min, confidence_max: Confidence range

    Returns matching images.
    """
    try:
        query = request.args.get("query", "").lower()
        score_min = request.args.get("score_min", 0.0, type=float)
        score_max = request.args.get("score_max", 1.0, type=float)
        confidence_min = request.args.get("confidence_min", 0.0, type=float)
        confidence_max = request.args.get("confidence_max", 1.0, type=float)

        all_images = get_all_images()

        results = []
        for img in all_images:
            # Name filter
            if query and query not in img["filename"].lower():
                continue

            # Score filter
            if not (score_min <= img["score"] <= score_max):
                continue

            # Confidence filter
            if not (confidence_min <= img["confidence"] <= confidence_max):
                continue

            results.append(
                {
                    "filename": img["filename"],
                    "score": round(img["score"], 3),
                    "file": f"{img['filename']}?score={round(img['score'],3)}",
                    "confidence": round(img["confidence"], 3),
                    "comparison_count": img["comparison_count"],
                }
            )

        return jsonify({"results": results, "count": len(results)})
    except Exception as e:
        logger.error(f"Error searching images: {e}")
        return jsonify({"error": str(e)}), 500


@gallery_bp.route("/history/<path:filename>", methods=["GET"])
def get_image_history(filename: str):
    """Get comparison history for a specific image from its JSON file."""
    try:
        img_path = find_image_path(filename)
        if not img_path:
            return jsonify({"error": "Image not found"}), 404

        json_path = img_path.with_suffix(".json")
        if not json_path.exists():
            return jsonify({"filename": filename, "history": []})

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        history = data.get("comparison_history", [])
        return jsonify({"filename": filename, "history": history})
    except Exception as e:
        logger.error(f"Error getting history for {filename}: {e}")
        return jsonify({"error": str(e)}), 500


def register_gallery_routes(app) -> None:
    """Register all gallery API v2 routes with Flask app."""
    app.register_blueprint(gallery_bp)
