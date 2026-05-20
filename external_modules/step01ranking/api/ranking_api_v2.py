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
from algorithm import merge_sort_ranker, state
from file_management.path_handler import sync_image_metadata_to_json
from shared.graph import crystal_graph

ranking_bp = Blueprint("ranking_v2", __name__, url_prefix="/api/v2/ranking")
logger: logging.Logger = logging.getLogger(__name__)


def _get_processor() -> Any:
    """Get the image processor from the current Flask app."""
    return getattr(current_app, "image_processor", None) or current_app.extensions.get(
        "image_processor"
    )


def _count_extreme_nodes_at_level(
    level: int, all_images_dict: dict[str, dict[str, Any]]
) -> int:
    """Count top/bottom graph nodes whose comparison count matches a level."""
    top_nodes = [n.filename for n in crystal_graph.get_all_nodes(only_top=True)]
    bottom_nodes = [n.filename for n in crystal_graph.get_all_nodes(only_bottom=True)]

    return len(
        [
            n
            for n in top_nodes
            if all_images_dict.get(n, {}).get("comparison_count", 0) == level
        ]
    ) + len(
        [
            n
            for n in bottom_nodes
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

    if active_nodes == 0:
        top_nodes = [n.filename for n in crystal_graph.get_all_nodes(only_top=True)]
        bottom_nodes = [
            n.filename for n in crystal_graph.get_all_nodes(only_bottom=True)
        ]
        extreme_counts = [
            all_images_dict.get(n, {}).get("comparison_count", 0)
            for n in top_nodes + bottom_nodes
        ]
        if extreme_counts:
            active_level = min(extreme_counts)
            active_nodes = _count_extreme_nodes_at_level(active_level, all_images_dict)

    next_level_count = _count_extreme_nodes_at_level(active_level + 1, all_images_dict)

    stats = crystal_graph.get_graph_stats()
    total_components = stats.get("total_components", 0)
    total_chains = stats.get("total_chains", 0)
    total_top_nodes = stats.get("top_nodes_count", 0)

    result = {
        "base_level": base_level,
        "active_level": active_level,
        "active_nodes": active_nodes,
        "next_level_count": next_level_count,
        "total_components": total_components,
        "total_chains": total_chains,
        "total_top_nodes": total_top_nodes,
    }
    logger.debug(f"[LEVEL] level_stats done: {time.time() - _s:.3f}s")
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

    _s = time.time()

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

    _s = time.time()

    level_stats = _get_level_progress_stats(all_images)

    _s = time.time()
    logger.info(f"[STATUS] TOTAL: {time.time() - _t:.3f}s")

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

        for attempt in range(max_retries):
            pair = merge_sort_ranker.select_pair_for_comparison(
                exclude_set=local_excluded_files_set, exclude_chains=excluded_chains_set
            )
            if not pair:
                break

            filename_a, filename_b = pair

            if (
                filename_a in local_excluded_files_set
                or filename_b in local_excluded_files_set
            ):
                local_excluded_files_set.update([filename_a, filename_b])
                continue

            if filename_a == filename_b:
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
                    if (
                        pair_meta.get("pair_type")
                        == merge_sort_ranker.PAIR_TYPE_REFINEMENT
                    ):
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
            node_a = crystal_graph.get_node(filename_a)
            node_b = crystal_graph.get_node(filename_b)

            try:
                chains_a = node_a.get_chain(only_main=True) if node_a else []
            except Exception as e:
                logger.warning(f"Failed to get chain for {filename_a}: {e}")
                chains_a = []
            height_a = chains_a[0].length if chains_a else 0

            try:
                chains_b = node_b.get_chain(only_main=True) if node_b else []
            except Exception as e:
                logger.warning(f"Failed to get chain for {filename_b}: {e}")
                chains_b = []

            height_b = chains_b[0].length if chains_b else 0

            comp_a = crystal_graph.get_component(node_id=filename_a)
            comp_b = crystal_graph.get_component(node_id=filename_b)

            comp_id_a = comp_a.id if comp_a else None
            comp_id_b = comp_b.id if comp_b else None
            comp_size_a = comp_a.size if comp_a else 1
            comp_size_b = comp_b.size if comp_b else 1

            # Count top/bottom nodes in each image's component (for debug)
            def _count_extremes(comp_proxy):
                if not comp_proxy:
                    return {"top": 0, "bottom": 0}
                top = sum(1 for m in comp_proxy.nodes if m.is_top())
                bottom = sum(1 for m in comp_proxy.nodes if m.is_bottom())
                return {"top": top, "bottom": bottom}

            extremes_a = _count_extremes(comp_a)
            extremes_b = _count_extremes(comp_b)

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
                        "is_top": node_a.is_top() if node_a else True,
                        "is_bottom": node_a.is_bottom() if node_a else True,
                    },
                    "right": {
                        "filename": data_b["filename"],
                        "score": round(data_b["score"], 4),
                        "confidence": round(data_b["confidence"], 4),
                        "comparison_count": data_b["comparison_count"],
                        "chain_length": height_b,
                        "component_size": comp_size_b,
                        "component_id": comp_id_b,
                        "is_top": node_b.is_top() if node_b else True,
                        "is_bottom": node_b.is_bottom() if node_b else True,
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
                        "refinement_details": pair_meta.get("refinement_details"),
                    },
                    "debug": {
                        "score_diff": round(abs(data_a["score"] - data_b["score"]), 4),
                        "left_extremes": extremes_a,
                        "right_extremes": extremes_b,
                        "max_graph_height": crystal_graph.get_graph_stats().get(
                            "longest_chain_depth", 0
                        ),
                        "total_components": level_stats["total_components"],
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

    try:
        processor = _get_processor()

        if processor:
            with processor.recent_lock:
                processor.recent_images.clear()
                processor.recent_chains.clear()
                state.clear_old_cache(force=True)
                crystal_graph.rebuild_from_database()
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
    if crystal_graph.is_cache_stale():
        logger.debug(
            "[SUBMIT] Graph cache is stale, rebuilding before applying comparison."
        )
        crystal_graph.rebuild_from_database()

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

        # Rebuild only if stale (apply_comparison keeps it fresh)
        if crystal_graph.is_cache_stale():
            crystal_graph.rebuild_from_database()

        stats = crystal_graph.get_graph_stats()

        # Pre-calculate heights to avoid O(N^3) chain loops for all 26,000 images
        node_to_height = {}
        for proxy in crystal_graph.get_all_chains():
            for node_proxy in proxy.nodes:
                filename = node_proxy.filename
                if (
                    filename not in node_to_height
                    or proxy.length > node_to_height[filename]
                ):
                    node_to_height[filename] = proxy.length

        # Prepare simplified nodes and edges for frontend visualization
        nodes = []
        all_nodes = crystal_graph.get_all_nodes()
        all_images_db = get_all_images()
        img_dict = {img["filename"]: img for img in all_images_db}

        for node in all_nodes:
            filename = node.filename
            img_data = img_dict.get(filename)
            comp = node.get_component()

            nodes.append(
                {
                    "id": filename,
                    "score": round(img_data["score"], 4) if img_data else 0.5,
                    "confidence": round(img_data["confidence"], 4) if img_data else 0.0,
                    "height": node_to_height.get(filename, 0),
                    "component": comp.id if comp else None,
                    "comparison_count": img_data["comparison_count"] if img_data else 0,
                    "is_top": node.is_top(),
                    "is_bottom": node.is_bottom(),
                }
            )

        edges = []
        for winner, loser in crystal_graph.get_all_links():
            edges.append(
                {
                    "source": winner.filename,
                    "target": loser.filename,
                    "weight": 1.0,
                }
            )

        all_components = crystal_graph.get_all_components()
        component_members = {
            comp.id: [n.filename for n in comp.nodes] for comp in all_components
        }

        # Serialize chains for frontend visualization
        chains = []
        try:
            for chain_proxy in crystal_graph.get_all_chains():
                comp = chain_proxy.get_component()
                chains.append(
                    {
                        "id": chain_proxy.id,
                        "component": comp.id if comp else None,
                        "nodes": [n.filename for n in chain_proxy.nodes],
                    }
                )
        except Exception as chain_err:
            logger.warning(f"Chain serialization skipped: {chain_err}")

        logger.info(
            f"Serialization complete. Returning {len(nodes)} nodes, {len(edges)} edges, {len(chains)} chains."
        )
        return jsonify(
            {
                "nodes": nodes,
                "edges": edges,
                "components": component_members,
                "chains": chains,
                "stats": {
                    "total_nodes": len(nodes),
                    "total_edges": len(edges),
                    "total_components": len(all_components),
                    "total_chains": stats.get("total_chains", 0),
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
