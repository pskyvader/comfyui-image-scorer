"""Gallery API - endpoints for viewing and filtering ranked images."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_root = Path(__file__).parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from flask import Blueprint, jsonify, request

from ..database_structure.comparisons_table import get_all_comparisons
from ..database_structure.images_table import get_all_images, get_image
import time

gallery_bp = Blueprint("gallery_v2", __name__, url_prefix="/api/v2/gallery")
from ...shared.logger import get_logger, ModuleLogger
logger: ModuleLogger = get_logger(__name__)


@gallery_bp.route("/images", methods=["GET"])
def list_images():
    _start = time.perf_counter()
    _start = time.perf_counter()
    page = max(request.args.get("page", 1, type=int), 1)
    per_page = request.args.get("per_page", 20, type=int)
    if per_page < 1 or per_page > 100:
        per_page = 20

    scored = get_all_images()
    score_min = request.args.get("score_min", 0.0, type=float)
    score_max = request.args.get("score_max", 1.0, type=float)
    comparisons_min = request.args.get("comparisons_min", 0, type=int)
    comparisons_max = request.args.get("comparisons_max", 999999, type=int)
    search_mode = request.args.get("search_mode", "both").lower()
    tags_query = request.args.get("tags", "").strip().lower()
    search_tags = [tag.strip() for tag in tags_query.split(",")] if tags_query else []

    filtered_and = []
    filtered_or = []
    for img in scored:
        if not (score_min <= float(img["score"]) <= score_max):
            continue
        comp_count = int(img.get("comparison_count", 0))
        if not (comparisons_min <= comp_count <= comparisons_max):
            continue

        if not search_tags:
            filtered_and.append(img)
            continue

        img_tags = ((img.get("prompt_tags") or "") + " " + img["filename"]).lower()
        if all(tag in img_tags for tag in search_tags):
            filtered_and.append(img)
        elif search_mode != "and" and any(tag in img_tags for tag in search_tags):
            filtered_or.append(img)

    sort_by = request.args.get("sort", "score_desc")

    def sort_list(items: list[dict[str, Any]]) -> None:
        if sort_by == "score_asc":
            items.sort(key=lambda item: float(item["score"]))
        elif sort_by == "score_desc":
            items.sort(key=lambda item: float(item["score"]), reverse=True)
        elif sort_by == "comparisons_asc":
            items.sort(key=lambda item: int(item["comparison_count"]))
        elif sort_by == "comparisons_desc":
            items.sort(key=lambda item: int(item["comparison_count"]), reverse=True)
        elif sort_by in {"last_compared_desc", "newest"}:
            items.sort(
                key=lambda item: item.get("last_compared_at") or "", reverse=True
            )
        elif sort_by == "last_compared_asc":
            items.sort(key=lambda item: item.get("last_compared_at") or "9999-99-99")

    if search_tags:
        sort_list(filtered_and)
        sort_list(filtered_or)
        filtered = (
            filtered_or
            if search_mode == "or"
            else filtered_and if search_mode == "and" else filtered_and + filtered_or
        )
    else:
        filtered = filtered_and
        sort_list(filtered)

    total = len(filtered)
    offset = (page - 1) * per_page
    paginated = filtered[offset : offset + per_page]
    images = [
        {
            "filename": img["filename"],
            "score": round(float(img["score"]), 3),
            "file": f"{img['filename']}?score={round(float(img['score']), 3)}",
            "comparison_count": int(img.get("comparison_count", 0)),
            "prompt_tags": img.get("prompt_tags"),
        }
        for img in paginated
    ]
    result = jsonify(
        {
            "images": images,
            "total": total,
            "page": page,
            "per_page": per_page,
            "max_comparison_count": max(
                (int(img.get("comparison_count", 0)) for img in scored), default=0
            ),
        }
    )
    return result


@gallery_bp.route("/image/<path:filename>", methods=["GET"])
def get_image_info(filename: str):
    img = get_image(filename)
    if not img:
        result = jsonify({"error": "Image not found"}), 404
        return result
    result = jsonify(
        {
            "filename": img["filename"],
            "score": round(float(img["score"]), 4),
            "comparison_count": int(img.get("comparison_count", 0)),
            "last_compared_at": img.get("last_compared_at"),
            "prompt_tags": img.get("prompt_tags"),
        }
    )
    return result


@gallery_bp.route("/search", methods=["GET"])
def search_images():
    _start = time.perf_counter()
    _start = time.perf_counter()
    query = request.args.get("query", "").lower()
    score_min = request.args.get("score_min", 0.0, type=float)
    score_max = request.args.get("score_max", 1.0, type=float)

    results = []
    for img in get_all_images():
        if query and query not in img["filename"].lower():
            continue
        if not (score_min <= float(img["score"]) <= score_max):
            continue
        results.append(
            {
                "filename": img["filename"],
                "score": round(float(img["score"]), 3),
                "file": f"{img['filename']}?score={round(float(img['score']), 3)}",
                "comparison_count": int(img.get("comparison_count", 0)),
            }
        )
    result = jsonify({"results": results, "count": len(results)})

    result = result

    return result


@gallery_bp.route("/history/<path:filename>", methods=["GET"])
def get_image_history(filename: str):
    wins = []
    losses = []
    for comp in get_all_comparisons():
        if comp["filename_a"] != filename and comp["filename_b"] != filename:
            continue
        opponent = (
            comp["filename_b"] if comp["filename_a"] == filename else comp["filename_a"]
        )
        opponent_data = get_image(opponent)
        item = {
            "opponent": opponent,
            "opponent_score": float(opponent_data["score"]) if opponent_data else 0.5,
            "timestamp": comp["timestamp"],
        }
        if comp["winner"] == filename:
            wins.append(item)
        else:
            losses.append(item)

    result = jsonify(
        {
            "filename": filename,
            "wins": wins,
            "losses": losses,
            "total_comparisons": len(wins) + len(losses),
        }
    )
    return result


def register_gallery_routes(app) -> None:
    _start = time.perf_counter()
    _start = time.perf_counter()
    app.register_blueprint(gallery_bp)
