"""Ranking API v2 - new endpoints for ranking system."""

import sys
import time
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


def _count_extreme_nodes_at_level(
    level: int, all_images_dict: dict[str, dict[str, Any]]
) -> int:
    """Count top/bottom graph nodes whose comparison count matches a level."""
    return len(
        [
            n
            for n in crystal_graph._chain._top_nodes
            if all_images_dict.get(n, {}).get("comparison_count", 0) == level
        ]
    ) + len(
        [
            n
            for n in crystal_graph._chain._bottom_nodes
            if all_images_dict.get(n, {}).get("comparison_count", 0) == level
        ]
    )


def _get_level_progress_stats(
    all_images: list[dict[str, Any]], rebuild_graph: bool = False
) -> dict[str, int]:
    """Compute active/next level stats for comparison progress."""
    _s = time.time()
    if rebuild_graph:
        crystal_graph.rebuild_from_database()

    all_images_dict = {img["filename"]: img for img in all_images}
    comp_counts = [img["comparison_count"] for img in all_images]
    base_level = min(comp_counts) if comp_counts else 0

    active_level = base_level
    active_nodes = _count_extreme_nodes_at_level(active_level, all_images_dict)
    logger.debug(f"[LEVEL] count_extreme level={active_level}: {time.time()-_s:.3f}s"); _s = time.time()

    if active_nodes == 0:
        extreme_counts = [
            all_images_dict.get(n, {}).get("comparison_count", 0)
            for n in crystal_graph._chain._top_nodes + crystal_graph._chain._bottom_nodes
        ]
        if extreme_counts:
            active_level = min(extreme_counts)
            active_nodes = _count_extreme_nodes_at_level(active_level, all_images_dict)
        logger.debug(f"[LEVEL] fallback level={active_level}: {time.time()-_s:.3f}s"); _s = time.time()

    next_level_count = _count_extreme_nodes_at_level(active_level + 1, all_images_dict)
    logger.debug(f"[LEVEL] count_extreme level={active_level+1}: {time.time()-_s:.3f}s"); _s = time.time()

    _comp_s = time.time()
    total_components = len(crystal_graph._chain._component_members)
    logger.debug(f"[LEVEL] component_members: {time.time()-_comp_s:.4f}s")
    _chain_s = time.time()
    total_chains = crystal_graph._chain.get_min_chain_count()
    logger.debug(f"[LEVEL] min_chain_count (cached): {time.time()-_chain_s:.4f}s")
    _top_s = time.time()
    total_top_nodes = len(crystal_graph._chain._top_nodes)
    logger.debug(f"[LEVEL] top_nodes: {time.time()-_top_s:.4f}s")

    result = {
        "base_level": base_level,
        "active_level": active_level,
        "active_nodes": active_nodes,
        "next_level_count": next_level_count,
        "total_components": total_components,
        "total_chains": total_chains,
        "total_top_nodes": total_top_nodes,
    }
    logger.debug(f"[LEVEL] level_stats done: {time.time()-_s:.3f}s")
    return result


@ranking_bp.route("/status", methods=["GET"])
def get_status():
    """
    Get ranking system status.
    """
    _t = time.time()
    _s = _t
    all_images = get_all_images()
    total = len(all_images)
    logger.info(f"[STATUS] get_all_images: {time.time()-_s:.3f}s"); _s = time.time()

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
    ranked = len([img for img in all_images if img["comparison_count"] > min_comps])
    unranked = total - ranked
    total_comps = get_total_comparisons()

    avg_conf = (
        sum(img["confidence"] for img in all_images) / total if total > 0 else 0.0
    )

    min_images = int(config["ranking"]["lru_size"])
    skipped_count = get_skipped_comparison_count()
    logger.info(f"[STATUS] comp/rank/skip calcs: {time.time()-_s:.3f}s"); _s = time.time()

    level_stats = _get_level_progress_stats(all_images)
    logger.info(f"[STATUS] level_stats: {time.time()-_s:.3f}s"); _s = time.time()


    return jsonify(
        {
            "total_images": total,
            "ranked_images": ranked,
            "unranked_images": unranked,
            "total_comparisons": total_comps,
            "skipped_comparisons": skipped_count,
            "average_confidence": round(avg_conf, 3),
            "min_images": min_images,
            "current_target": level_stats["active_level"],
            "baseline_comparisons": level_stats["base_level"],
            "total_components": level_stats["total_components"],
            "total_chains": level_stats["total_chains"],
            "active_nodes": level_stats["active_nodes"],
            "next_level_count": level_stats["next_level_count"],
            "base_level": level_stats["base_level"],
        }
    )

    logger.info(f"[STATUS] TOTAL: {time.time()-_t:.3f}s")


@ranking_bp.route("/next-pair", methods=["GET"])
def get_next_pair():
    """
    Get next pair of images for comparison.
    """
    processor = _get_processor()
    excluded_files_set: set[str] = set()
    excluded_chains_set: set[tuple[str, ...]] = set()
    recent_files_ordered: list[str] = []
    recent_chains_ordered: list[tuple[str, ...]] = []

    if processor:
        with processor.recent_lock:
            recent_files_ordered = list(processor.recent_images)
            recent_chains_ordered = list(processor.recent_chains)
            excluded_files_set = set(recent_files_ordered)
            excluded_chains_set = set(recent_chains_ordered)

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
    exclusion_variants: list[set[str]] = []
    for variant in (
        set(recent_files_ordered),
        set(recent_files_ordered[-4:]),
        set(recent_files_ordered[-2:]),
        set(),
    ):
        if variant not in exclusion_variants:
            exclusion_variants.append(variant)

    for variant_index, base_excluded_set in enumerate(exclusion_variants, start=1):
        local_excluded_files_set = set(base_excluded_set)
        logger.debug(
            f"[NEXT-PAIR] Trying exclusion variant {variant_index}/{len(exclusion_variants)} with {len(local_excluded_files_set)} images"
        )

        for attempt in range(max_retries):
            pair = merge_sort_ranker.select_pair_for_comparison(
                exclude_set=local_excluded_files_set,
                exclude_chains=excluded_chains_set
            )
            if not pair:
                logger.debug(
                    f"[NEXT-PAIR] No pair found after {attempt+1} attempts for variant {variant_index}"
                )
                break

            filename_a, filename_b = pair

            if (
                filename_a in local_excluded_files_set
                or filename_b in local_excluded_files_set
            ):
                logger.debug(
                    f"[NEXT-PAIR] Retry: {filename_a} or {filename_b} in excluded set"
                )
                local_excluded_files_set.update([filename_a, filename_b])
                continue

            if filename_a == filename_b:
                logger.debug(f"[NEXT-PAIR] Retry: same image {filename_a}")
                local_excluded_files_set.add(filename_a)
                continue

            data_a = get_img_data(filename_a)
            data_b = get_img_data(filename_b)

            if not data_a or not data_b or data_a["filename"] == data_b["filename"]:
                local_excluded_files_set.update([filename_a, filename_b])
                continue

            if processor:
                with processor.recent_lock:
                    # Update images cache
                    processor.recent_images.append(filename_a)
                    processor.recent_images.append(filename_b)
                    if len(processor.recent_images) >= lru_size:
                        for _ in range(int(len(processor.recent_images) * 0.75)):
                            processor.recent_images.pop()

                    # Update chains cache if it was a refinement pair
                    pair_meta = merge_sort_ranker.get_last_pair_metadata()
                    if pair_meta.get("pair_type") == merge_sort_ranker.PAIR_TYPE_REFINEMENT:
                        chain_a = pair_meta.get("chain_a")
                        chain_b = pair_meta.get("chain_b")
                        if chain_a:
                            processor.recent_chains.append(chain_a)
                        if chain_b:
                            processor.recent_chains.append(chain_b)
                        
                        if len(processor.recent_chains) >= lru_size:
                            for _ in range(int(len(processor.recent_chains) * 0.75)):
                                processor.recent_chains.pop()

            # Get pair selection metadata from the ranker
            pair_meta = merge_sort_ranker.get_last_pair_metadata()

            # Calculate global stats for the status line
            all_images_for_stats = get_all_images()
            total_images_stats = len(all_images_for_stats)
            total_comparisons_stats = get_total_comparisons()

            comp_counts = [img["comparison_count"] for img in all_images_for_stats]
            min_comps = min(comp_counts) if comp_counts else 0
            level_count = len([c for c in comp_counts if c > min_comps])
            skipped_for_stats = get_skipped_comparison_count()
            level_stats = _get_level_progress_stats(all_images_for_stats)

            # Compute chain_length (height) and component_size for each image
            height_a = crystal_graph._chain._chain_length.get(filename_a, 0)
            height_b = crystal_graph._chain._chain_length.get(filename_b, 0)
            comp_id_a = crystal_graph._chain._node_component.get(filename_a)
            comp_id_b = crystal_graph._chain._node_component.get(filename_b)
            comp_size_a = (
                len(crystal_graph._chain._component_members.get(comp_id_a, []))
                if comp_id_a is not None
                else 1
            )
            comp_size_b = (
                len(crystal_graph._chain._component_members.get(comp_id_b, []))
                if comp_id_b is not None
                else 1
            )

            # Count top/bottom nodes in each image's component (for debug)
            def _count_extremes(comp_id):
                if comp_id is None:
                    return {"top": 0, "bottom": 0}
                members = crystal_graph._chain._component_members.get(comp_id, [])
                top = sum(1 for m in members if not crystal_graph._chain._better_than.get(m, []))
                bottom = sum(1 for m in members if not crystal_graph._chain._worse_than.get(m, []))
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
                        "is_top": len(crystal_graph._chain._better_than.get(filename_a, [])) == 0,
                        "is_bottom": len(crystal_graph._chain._worse_than.get(filename_a, [])) == 0,
                    },
                    "right": {
                        "filename": data_b["filename"],
                        "score": round(data_b["score"], 4),
                        "confidence": round(data_b["confidence"], 4),
                        "comparison_count": data_b["comparison_count"],
                        "chain_length": height_b,
                        "component_size": comp_size_b,
                        "component_id": comp_id_b,
                        "is_top": len(crystal_graph._chain._better_than.get(filename_b, [])) == 0,
                        "is_bottom": len(crystal_graph._chain._worse_than.get(filename_b, [])) == 0,
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
                        "max_graph_height": max(crystal_graph._chain._chain_length.values()) if crystal_graph._chain._chain_length else 0,
                        "total_components": len(crystal_graph._chain._component_members),
                    },
                    "global_stats": {
                        "total_images": total_images_stats,
                        "total_comparisons": total_comparisons_stats,
                        "skipped_comparisons": skipped_for_stats,
                        "level_count": level_count,
                        "total_components": level_stats["total_components"],
                        "total_chains": level_stats["total_chains"],
                        "active_nodes": level_stats["active_nodes"],
                        "next_level_count": level_stats["next_level_count"],
                        "target_level": level_stats["active_level"],
                        "base_level": level_stats["base_level"],
                    },
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
                image_processor.recent_images.clear(); image_processor.recent_chains.clear()
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

        logger.info("Generating graph data...")

        # Rebuild only if stale (apply_comparison keeps it fresh)
        if crystal_graph.is_cache_stale():
            crystal_graph.rebuild_from_database()
            logger.info("Graph was stale, rebuilt from database")
        else:
            logger.info("Graph is fresh, using in-memory data")

        logger.info(
            f"Graph data: {len(crystal_graph._images)} nodes, {len(crystal_graph._comparisons)} comparisons, "
            f"{len(crystal_graph._chain._component_members)} components"
        )

        logger.info(f"Components: {len(crystal_graph._chain._component_members)}")

        # Prepare simplified nodes and edges for frontend visualization
        nodes = []
        for filename, img_data in crystal_graph._images.items():
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
                    "height": crystal_graph._chain._chain_length.get(filename, 0),
                    "component": crystal_graph._chain._node_component.get(filename),
                    "comparison_count": img_data["comparison_count"],
                    "is_top": len(crystal_graph._chain._better_than.get(filename, [])) == 0,
                    "is_bottom": len(crystal_graph._chain._worse_than.get(filename, [])) == 0,
                }
            )

        edges = []
        for comp in crystal_graph._comparisons:
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
                "components": crystal_graph._chain._component_members,
                "stats": {
                    "total_nodes": len(nodes),
                    "total_edges": len(edges),
                    "total_components": len(crystal_graph._chain._component_members),
                    "total_chains": crystal_graph._chain.get_min_chain_count(),
                },
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
