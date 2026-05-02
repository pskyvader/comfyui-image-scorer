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
from algorithm import merge_sort_ranker
from file_management.path_handler import sync_image_metadata_to_json
from shared.graph import crystal_graph


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
    - ranked_images: Images that have reached the next comparison level
    - unranked_images: Images at the minimum comparison level
    - total_comparisons: Total comparisons made
    - average_confidence: Average confidence across images
    - current_target: The comparison count we are currently working towards
    """
    all_images = get_all_images()
    total = len(all_images)
    
    if total == 0:
        return jsonify({
            "total_images": 0,
            "ranked_images": 0,
            "unranked_images": 0,
            "total_comparisons": 0,
            "skipped_comparisons": 0,
            "average_confidence": 0.0,
            "min_images": int(config["ranking"]["lru_size"]),
            "current_target": 1
        })

    comp_counts = [img["comparison_count"] for img in all_images]
    min_comps = min(comp_counts)
    
    # Ranked is now defined as images that have moved past the current baseline
    ranked = len([img for img in all_images if img["comparison_count"] > min_comps])
    
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
            "current_target": min_comps + 1,
            "baseline_comparisons": min_comps
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
        pair = merge_sort_ranker.select_pair_for_comparison(exclude_set=excluded_files_set)
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
                "collapsable": merge_sort_ranker.is_collapsable_pair(filename_a, filename_b),
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

    success = merge_sort_ranker.record_comparison(filename_a, filename_b, winner)

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


@ranking_bp.route("/graph-data", methods=["GET"])
def get_graph_data():
    """
    Get current comparison graph data for visualization.
    """
    try:
        from shared.graph import crystal_graph
        from database.images_table import get_all_images
        from database.comparisons_table import get_all_comparisons
        logger.info("Generating graph data...")
        
        # Check database directly
        raw_images = get_all_images()
        raw_comparisons = get_all_comparisons()
        logger.info(f"DB check: {len(raw_images)} raw images, {len(raw_comparisons)} raw comparisons")
        
        # Use the global crystal_graph instance
        crystal_graph.build_from_database()
        
        logger.info(f"Graph built: {len(crystal_graph.images)} nodes and {len(crystal_graph.comparisons)} comparisons")
        logger.info(f"Components: {crystal_graph.chain_members_by_component}")
        
        # Prepare simplified nodes and edges for frontend visualization
        nodes = []
        for filename, img_data in crystal_graph.images.items():
            nodes.append({
                "id": filename,
                "score": round(img_data["score"], 4) if img_data and img_data.get("score") is not None else 0.5,
                "confidence": round(img_data["confidence"], 4) if img_data and img_data.get("confidence") is not None else 0.0,
                "height": crystal_graph.height.get(filename, 0),
                "component": crystal_graph.component_by_filename.get(filename)
            })
            
        edges = []
        for comp in crystal_graph.comparisons:
            winner = comp["winner"]
            filename_a = comp["filename_a"]
            filename_b = comp["filename_b"]
            loser = filename_b if winner == filename_a else filename_a
            edges.append({
                "source": winner,
                "target": loser,
                "weight": float(comp.get("weight", 1.0) or 1.0)
            })
            
        logger.info(f"Serialization complete. Returning {len(nodes)} nodes and {len(edges)} edges.")
        return jsonify({
            "nodes": nodes,
            "edges": edges,
            "components": crystal_graph.chain_members_by_component
        })
    except Exception as e:
        logger.error(f"Error in get_graph_data: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


def register_ranking_routes(app) -> None:
    """Register all ranking API v2 routes with Flask app."""
    app.register_blueprint(ranking_bp)


@ranking_bp.route("/sync-all", methods=["POST"])
def sync_all_to_json():
    """Sync all current DB scores/confidence to their JSON files (Backup)."""
    try:

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
