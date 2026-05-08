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
from database.images_table import (
    get_all_images,
    get_image_count,
    get_image as get_img_data,
)
from shared.config import config
from database.comparisons_table import (
    get_total_comparisons,
    get_skipped_comparison_count,
)
from algorithm import merge_sort_ranker
from file_management.path_handler import sync_image_metadata_to_json
from shared.graph import crystal_graph

ranking_bp = Blueprint("ranking_v2", __name__, url_prefix="/api/v2/ranking")
logger = logging.getLogger(__name__)


def _get_processor() -> Any:
    """Get the image processor from the current Flask app."""
    return getattr(current_app, "image_processor", None) or current_app.extensions.get(
        "image_processor"
    )


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
        return jsonify(
            {
                "total_images": 0,
                "ranked_images": 0,
                "unranked_images": 0,
                "total_comparisons": 0,
                "skipped_comparisons": 0,
                "average_confidence": 0.0,
                "min_images": int(config["ranking"]["lru_size"]),
                "current_target": 1,
            }
        )

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
            "baseline_comparisons": min_comps,
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
        pair = merge_sort_ranker.select_pair_for_comparison(
            exclude_set=excluded_files_set
        )
        if not pair:
            logger.debug(f"[NEXT-PAIR] No pair found after {attempt+1} attempts")
            return "", 204

        filename_a, filename_b = pair

        if filename_a in excluded_files_set or filename_b in excluded_files_set:
            logger.debug(
                f"[NEXT-PAIR] Retry: {filename_a} or {filename_b} in excluded set"
            )
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
                if len(processor.recent_images) >= lru_size:
                    for _ in range(int(len(processor.recent_images) * 0.75)):
                        processor.recent_images.pop()

        # Get pair selection metadata from the ranker
        pair_meta = merge_sort_ranker.get_last_pair_metadata()
        
        # Calculate global stats for the status line
        all_images_for_stats = get_all_images()
        total_images_stats = len(all_images_for_stats)
        total_comparisons_stats = get_total_comparisons()
        
        comp_counts = [img["comparison_count"] for img in all_images_for_stats]
        min_comps = min(comp_counts) if comp_counts else 0
        level_count = len([c for c in comp_counts if c > min_comps])
        target_level = min_comps + 1

        # Compute chain_length (height) and component_size for each image
        height_a = crystal_graph.height.get(filename_a, 0)
        height_b = crystal_graph.height.get(filename_b, 0)
        comp_id_a = crystal_graph.component_by_filename.get(filename_a)
        comp_id_b = crystal_graph.component_by_filename.get(filename_b)
        comp_size_a = len(crystal_graph.chain_members_by_component.get(comp_id_a, [])) if comp_id_a is not None else 1
        comp_size_b = len(crystal_graph.chain_members_by_component.get(comp_id_b, [])) if comp_id_b is not None else 1

        # Count top/bottom nodes in each image's component (for debug)
        def _count_extremes(comp_id):
            if comp_id is None:
                return {"top": 0, "bottom": 0}
            members = crystal_graph.chain_members_by_component.get(comp_id, [])
            top = sum(1 for m in members if not crystal_graph.better.get(m, []))
            bottom = sum(1 for m in members if not crystal_graph.worse.get(m, []))
            return {"top": top, "bottom": bottom}

        extremes_a = _count_extremes(comp_id_a)
        extremes_b = _count_extremes(comp_id_b)

        return jsonify(
            {
                "left": {
                    "filename": data_a["filename"],
                    "score": round(data_a["score"], 4),
                    "confidence": round(data_a["confidence"], 4),
                    "comparison_count": data_a["comparison_count"],
                    "chain_length": height_a,
                    "component_size": comp_size_a,
                    "component_id": comp_id_a,
                    "is_top": len(crystal_graph.better.get(filename_a, [])) == 0,
                    "is_bottom": len(crystal_graph.worse.get(filename_a, [])) == 0,
                },
                "right": {
                    "filename": data_b["filename"],
                    "score": round(data_b["score"], 4),
                    "confidence": round(data_b["confidence"], 4),
                    "comparison_count": data_b["comparison_count"],
                    "chain_length": height_b,
                    "component_size": comp_size_b,
                    "component_id": comp_id_b,
                    "is_top": len(crystal_graph.better.get(filename_b, [])) == 0,
                    "is_bottom": len(crystal_graph.worse.get(filename_b, [])) == 0,
                },
                "collapsable": merge_sort_ranker.is_collapsable_pair(
                    filename_a, filename_b
                ),
                "same_component": {
                    "id": comp_id_a if comp_id_a == comp_id_b else None,
                    "size": comp_size_a if comp_id_a == comp_id_b else None,
                },
                "pair_meta": {
                    "pair_type": pair_meta.get("pair_type", "unknown"),
                    "chain_level": pair_meta.get("chain_level", -1),
                    "component_size_group": pair_meta.get("component_size", -1),
                    "left_component_size": comp_size_a,
                    "right_component_size": comp_size_b,
                    "left_comp_count": pair_meta.get("left_comp_count", 0),
                    "right_comp_count": pair_meta.get("right_comp_count", 0),
                },
                "debug": {
                    "score_diff": round(abs(data_a["score"] - data_b["score"]), 4),
                    "left_extremes": extremes_a,
                    "right_extremes": extremes_b,
                    "max_graph_height": crystal_graph.get_max_height(),
                    "total_components": len(crystal_graph.chain_members_by_component),
                },
                "global_stats": {
                    "total_images": total_images_stats,
                    "total_comparisons": total_comparisons_stats,
                    "level_count": level_count,
                    "target_level": target_level
                }
            }
        )

    return "", 204


@ranking_bp.route("/reset", methods=["POST"])
def reset_ranking_queue():
    """
    Reset the comparison queue state.
    """
    logger.info("[RESET] Client requested ranking queue reset.")
    try:
        from server import image_processor
        if image_processor:
            with image_processor.recent_lock:
                image_processor.recent_images.clear()
                logger.info("[RESET] Cleared processor recent images LRU queue.")
        return jsonify({"status": "success", "message": "Ranking queue reset."})
    except Exception as e:
        logger.error(f"[RESET] Failed to reset queue: {e}")
        return jsonify({"error": "Failed to reset queue"}), 500


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
        logger.error(
            f"[SUBMIT] Failed to record comparison: {filename_a} vs {filename_b}, winner: {winner}"
        )
        return jsonify({"error": "Failed to record comparison"}), 500

    # Submitted images should remain in the LRU cache to prevent immediate re-selection.
    # The cache automatically manages removal of oldest items when the limit is reached.

    data_a = get_img_data(filename_a)
    data_b = get_img_data(filename_b)

    if data_a is None or data_b is None:
        logger.error(
            f"[SUBMIT] Image not found after comparison: {filename_a} or {filename_b}"
        )
        return jsonify({"error": "Image not found"}), 404

    logger.info(
        f"[SUBMIT] Comparison successful. A: {filename_a} (score: {data_a['score']:.3f}), B: {filename_b} (score: {data_b['score']:.3f}), Winner: {winner}"
    )

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
        logger.info(
            f"DB check: {len(raw_images)} raw images, {len(raw_comparisons)} raw comparisons"
        )

        # Use the global crystal_graph instance
        crystal_graph.build_from_database()

        logger.info(
            f"Graph built: {len(crystal_graph.images)} nodes and {len(crystal_graph.comparisons)} comparisons"
        )
        logger.info(f"Components: {crystal_graph.chain_members_by_component}")

        # Prepare simplified nodes and edges for frontend visualization
        nodes = []
        for filename, img_data in crystal_graph.images.items():
            nodes.append(
                {
                    "id": filename,
                    "score": (
                        round(img_data["score"], 4)
                        if img_data and img_data.get("score") is not None
                        else 0.5
                    ),
                    "confidence": (
                        round(img_data["confidence"], 4)
                        if img_data and img_data.get("confidence") is not None
                        else 0.0
                    ),
                    "height": crystal_graph.height.get(filename, 0),
                    "component": crystal_graph.component_by_filename.get(filename),
                    "comparison_count": img_data["comparison_count"],
                    "is_top": len(crystal_graph.better.get(filename, [])) == 0,
                    "is_bottom": len(crystal_graph.worse.get(filename, [])) == 0,
                }
            )

        edges = []
        for comp in crystal_graph.comparisons:
            winner = comp["winner"]
            filename_a = comp["filename_a"]
            filename_b = comp["filename_b"]
            loser = filename_b if winner == filename_a else filename_a
            edges.append(
                {
                    "source": winner,
                    "target": loser,
                    "weight": float(comp.get("weight", 1.0) or 1.0),
                }
            )

        logger.info(
            f"Serialization complete. Returning {len(nodes)} nodes and {len(edges)} edges."
        )
        return jsonify(
            {
                "nodes": nodes,
                "edges": edges,
                "components": crystal_graph.chain_members_by_component,
            }
        )
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

        # Fetch all comparisons once for efficiency
        from database.comparisons_table import get_all_comparisons

        all_comparisons = get_all_comparisons()

        for img in images:
            success = sync_image_metadata_to_json(
                img["filename"],
                img["score"],
                img["confidence"],
                img["comparison_count"],
                all_comparisons=all_comparisons,
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
