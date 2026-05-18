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
from database.images_table import (
    get_all_images,
    get_image,
)
from database.comparisons_table import get_all_comparisons

gallery_bp = Blueprint("gallery_v2", __name__, url_prefix="/api/v2/gallery")
logger = logging.getLogger(__name__)


@gallery_bp.route("/images", methods=["GET"])
def list_images():
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
    search_mode = request.args.get("search_mode", "both").lower()
    tags_query = request.args.get("tags", "").strip().lower()
    search_tags: list[str] = (
        [t.strip() for t in tags_query.split(",")] if tags_query else []
    )

    filtered_and = []
    filtered_or = []

    for img in scored:
        if not (score_min <= img["score"] <= score_max):
            continue
        if not (confidence_min <= img["confidence"] <= confidence_max):
            continue
        comp_count = img["comparison_count"]
        if not (comparisons_min <= comp_count <= comparisons_max):
            continue

        if not search_tags:
            filtered_and.append(img)
            continue

        img_tags = (img["prompt_tags"] or "").lower()

        all_match = True
        for t in search_tags:
            if t not in img_tags:
                all_match = False
                break

        if all_match:
            filtered_and.append(img)
        elif search_mode != "and":
            any_match = False
            for t in search_tags:
                if t in img_tags:
                    any_match = True
                    break
            if any_match:
                filtered_or.append(img)

    sort_by = request.args.get("sort", "score_desc")

    def sort_list(lst):
        if sort_by == "score_asc":
            lst.sort(key=lambda x: x["score"])
        elif sort_by == "score_desc":
            lst.sort(key=lambda x: x["score"], reverse=True)
        elif sort_by == "confidence_asc":
            lst.sort(key=lambda x: x["confidence"])
        elif sort_by == "confidence_desc":
            lst.sort(key=lambda x: x["confidence"], reverse=True)
        elif sort_by == "comparisons_asc":
            lst.sort(key=lambda x: x["comparison_count"])
        elif sort_by == "comparisons_desc":
            lst.sort(key=lambda x: x["comparison_count"], reverse=True)
        elif sort_by == "last_compared_desc" or sort_by == "newest":
            lst.sort(key=lambda x: x["last_compared_at"] or "", reverse=True)
        elif sort_by == "last_compared_asc":
            lst.sort(key=lambda x: x["last_compared_at"] or "9999-99-99")

    if search_tags:
        sort_list(filtered_and)
        sort_list(filtered_or)
        if search_mode == "or":
            filtered = filtered_or
        elif search_mode == "and":
            filtered = filtered_and
        else:
            filtered = filtered_and + filtered_or
    else:
        filtered = filtered_and
        sort_list(filtered)

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
            "prompt_tags": img["prompt_tags"],
        }
        for img in paginated
    ]

    return jsonify(
        {
            "images": images,
            "total": total,
            "page": page,
            "per_page": per_page,
            "max_comparison_count": max(
                (img["comparison_count"] for img in scored), default=0
            ),
        }
    )


@gallery_bp.route("/image/<path:filename>", methods=["GET"])
def get_image_info(filename: str):
    img = get_image(filename)
    if not img:
        return jsonify({"error": "Image not found"}), 404

    return jsonify(
        {
            "filename": img["filename"],
            "score": round(img["score"], 4),
            "confidence": round(img["confidence"], 4),
            "comparison_count": img["comparison_count"],
            "last_compared_at": img["last_compared_at"],
            "prompt_tags": img["prompt_tags"],
        }
    )


@gallery_bp.route("/search", methods=["GET"])
def search_images():
    query = request.args.get("query", "").lower()
    score_min = request.args.get("score_min", 0.0, type=float)
    score_max = request.args.get("score_max", 1.0, type=float)
    confidence_min = request.args.get("confidence_min", 0.0, type=float)
    confidence_max = request.args.get("confidence_max", 1.0, type=float)

    all_images = get_all_images()

    results = []
    for img in all_images:
        if query and query not in img["filename"].lower():
            continue

        if not (score_min <= img["score"] <= score_max):
            continue

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


@gallery_bp.route("/history/<path:filename>", methods=["GET"])
def get_image_history(filename: str):
    all_comparisons = get_all_comparisons()
    wins = []
    losses = []

    for comp in all_comparisons:
        if comp["filename_a"] != filename and comp["filename_b"] != filename:
            continue
        is_winner = comp["winner"] == filename
        opponent = comp["filename_b"] if comp["filename_a"] == filename else comp["filename_a"]
        opponent_data = get_image(opponent)
        item = {
            "opponent": opponent,
            "opponent_score": opponent_data["score"],
            "timestamp": comp["timestamp"],
        }
        if is_winner:
            wins.append(item)
        else:
            losses.append(item)

    return jsonify(
        {
            "filename": filename,
            "wins": wins,
            "losses": losses,
            "total_comparisons": len(wins) + len(losses),
        }
    )


def register_gallery_routes(app) -> None:
    """Register all gallery API v2 routes with Flask app."""
    app.register_blueprint(gallery_bp)
