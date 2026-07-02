"""Ranking API v2 endpoints."""

from __future__ import annotations

import time
import sys
from pathlib import Path
from typing import Any

_root: Path = Path(__file__).parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from flask import Blueprint, current_app, jsonify, request

from ..comparison.algorithm import (
    merge_sort_ranker,
    state,
    graph_helpers,
    comparison_recorder,
)
from ..comparison.algorithm.pair_active import _stable_seed_pool
from ..database_structure.comparisons_table import (
    get_all_comparisons,
    get_skipped_comparison_count,
    get_total_comparisons,
)
from ..database_structure.images_table import (
    get_all_images,
    get_image_count,
)
from ..database_structure.path_handler import sync_image_metadata_to_json
from ...shared.config import config
from ...shared.graph.crystal_graph import crystal_graph

ranking_bp = Blueprint("ranking_v2", __name__, url_prefix="/api/v2/ranking")
from ...shared.logger import get_logger, ModuleLogger
logger: ModuleLogger = get_logger(__name__)


def _get_processor():
    return getattr(current_app, "image_processor", None) or current_app.extensions.get(
        "image_processor"
    )


def _get_level_progress_stats(
    all_images: list[dict[str, Any]],
) -> dict[str, int]:
    _start = time.perf_counter()
    comp_counts = [int(img["comparison_count"]) for img in all_images]
    base_level = min(comp_counts)
    active_nodes = sum(1 for count in comp_counts if count == base_level)
    next_level_count = sum(1 for count in comp_counts if count == base_level + 1)
    stats = crystal_graph.get_graph_stats()
    result = {
        "base_level": base_level,
        "current_target": base_level + 1,
        "active_nodes": active_nodes,
        "next_level_count": next_level_count,
        "total_components": stats["total_components"],
        "total_chains": stats["total_chains"],
    }

    return result


def _node_payload(filename: str, img_data: dict[str, Any]) -> dict[str, Any]:
    node = crystal_graph.get_node(filename)
    comp = crystal_graph.get_component(node_id=filename)
    chain_length = crystal_graph.get_node_chain_length(filename)

    left_extremes = {"top": 0, "bottom": 0}
    if comp:
        left_extremes = {
            "top": sum(1 for member in comp.nodes if member.is_top()),
            "bottom": sum(1 for member in comp.nodes if member.is_bottom()),
        }

    result = {
        "filename": img_data["filename"],
        "score": round(float(img_data["score"]), 4),
        "comparison_count": int(img_data["comparison_count"]),
        "chain_length": chain_length,
        "component_size": comp.size if comp else 0,
        "component_id": comp.id if comp else None,
        "is_top": node.is_top() if node else False,
        "is_bottom": node.is_bottom() if node else False,
        "_extremes": left_extremes,
    }
    if int(result["comparison_count"]) > 0 and result["chain_length"] == 0:
        raise RuntimeError(
            "a node can't have chain length 0 and comparison count > 0 at the same time"
        )
    return result


@ranking_bp.route("/config", methods=["GET"])
def get_ranking_config():
    _start = time.perf_counter()
    ranking_conf = config["ranking"]
    result = jsonify(
        {
            "reserve_count": int(ranking_conf["reserve_count"]),
            "parallel_requests": bool(ranking_conf["parallel_requests"]),
            "timeout_ms": int(ranking_conf["timeout_ms"]),
            "seed_size": int(ranking_conf["seed_size"]),
            "seed_target_comparisons": int(ranking_conf["seed_target_comparisons"]),
            "insertion_target_comparisons": int(
                ranking_conf["insertion_target_comparisons"]
            ),
        }
    )

    return result


@ranking_bp.route("/status", methods=["GET"])
def get_status():
    _start = time.perf_counter()
    all_images = get_all_images()
    total = len(all_images)
    if total == 0:
        result = jsonify(
            {
                "total_images": 0,
                "ranked_images": 0,
                "unranked_images": 0,
                "total_comparisons": 0,
                "skipped_comparisons": 0,
                "min_images": 2,
                "current_target": 1,
                "baseline_comparisons": 0,
                "total_components": 0,
                "total_chains": 0,
                "active_nodes": 0,
                "next_level_count": 0,
                "base_level": 0,
            }
        )

        return result

    level_stats = _get_level_progress_stats(all_images)
    ranked = sum(1 for img in all_images if int(img["comparison_count"]) > 0)
    result = jsonify(
        {
            "total_images": total,
            "ranked_images": ranked,
            "unranked_images": total - ranked,
            "total_comparisons": get_total_comparisons(),
            "skipped_comparisons": get_skipped_comparison_count(),
            "min_images": 2,
            "current_target": level_stats["current_target"],
            "baseline_comparisons": level_stats["base_level"],
            "total_components": level_stats["total_components"],
            "total_chains": level_stats["total_chains"],
            "active_nodes": level_stats["active_nodes"],
            "next_level_count": level_stats["next_level_count"],
            "base_level": level_stats["base_level"],
        }
    )

    return result


@ranking_bp.route("/next-pair", methods=["GET"])
def get_next_pair():
    _start = time.perf_counter()
    processor = _get_processor()
    recent_files_ordered: list[str] = []
    if processor:
        with processor.recent_lock:
            recent_files_ordered = list(processor.recent_images)

    total_images = get_image_count()
    if total_images < 2:
        result = (
            jsonify(
                {
                    "error": "Not Enough Images",
                    "message": "At least two valid images are required to start ranking.",
                }
            ),
            400,
        )
        return result

    full_exclude = set(recent_files_ordered)
    pair = merge_sort_ranker.select_pair_for_comparison(
        exclude_set=full_exclude,
    )
    if not pair:
        result = "", 204

        return result

    filename_a, filename_b = pair
    data_a = state.get_cached_image(filename_a)
    data_b = state.get_cached_image(filename_b)
    if not data_a or not data_b or data_a["filename"] == data_b["filename"]:
        result = "", 204

        return result

    if processor:
        with processor.recent_lock:
            processor.recent_images.append(filename_a)
            processor.recent_images.append(filename_b)

    all_images = get_all_images()
    seed_set = set(_stable_seed_pool(all_images))
    pair_meta = state.get_last_pair_metadata()
    level_stats = _get_level_progress_stats(all_images)
    level_count = sum(
        1
        for img in all_images
        if int(img["comparison_count"]) > level_stats["base_level"]
    )

    left = _node_payload(filename_a, data_a)
    right = _node_payload(filename_b, data_b)

    left_data = {
        "filename": left["filename"],
        "score": left["score"],
        "comparison_count": left["comparison_count"],
        "chain_length": left["chain_length"],
        "component_size": left["component_size"],
        "component_id": left["component_id"],
        "is_top": left["is_top"],
        "is_bottom": left["is_bottom"],
        "is_seed": filename_a in seed_set,
    }
    right_data = {
        "filename": right["filename"],
        "score": right["score"],
        "comparison_count": right["comparison_count"],
        "chain_length": right["chain_length"],
        "component_size": right["component_size"],
        "component_id": right["component_id"],
        "is_top": right["is_top"],
        "is_bottom": right["is_bottom"],
        "is_seed": filename_b in seed_set,
    }
    component_data = {"id": None, "size": None}
    if left["component_id"] == right["component_id"]:
        component_data["id"] = left["component_id"]
        component_data["size"] = left["component_size"]

    pair_data = {
        "pair_type": pair_meta.get("pair_type", "unknown"),
        "left_component_size": left["component_size"],
        "right_component_size": right["component_size"],
        "left_comp_count": pair_meta.get("left_comp_count", 0),
        "right_comp_count": pair_meta.get("right_comp_count", 0),
        "refinement_details": pair_meta.get("refinement_details", {}),
    }

    debug_data = {
        "score_diff": round(abs(left["score"] - right["score"]), 4),
        "left_extremes": left["_extremes"],
        "right_extremes": right["_extremes"],
        "max_graph_height": crystal_graph.get_graph_stats()["longest_chain_depth"],
        "total_components": level_stats["total_components"],
    }

    stats_data = {
        "total_images": len(all_images),
        "total_comparisons": get_total_comparisons(),
        "skipped_comparisons": get_skipped_comparison_count(),
        "level_count": level_count,
        "total_components": level_stats["total_components"],
        "total_chains": level_stats["total_chains"],
        "active_nodes": level_stats["active_nodes"],
        "next_level_count": level_stats["next_level_count"],
        "target_level": level_stats["current_target"],
        "base_level": level_stats["base_level"],
    }

    response_data = {
        "left": left_data,
        "right": right_data,
        "collapsable": graph_helpers.is_collapsable_pair(filename_a, filename_b),
        "same_component": component_data,
        "pair_meta": pair_data,
        "debug": debug_data,
        "global_stats": stats_data,
    }

    result = jsonify(response_data)
    return result


@ranking_bp.route("/reset", methods=["POST"])
def reset_ranking_queue():
    _start = time.perf_counter()
    processor = _get_processor()
    if processor:
        with processor.recent_lock:
            processor.clear_old_cache(force=True)
    result = jsonify({"status": "success", "message": "Ranking queue reset."})
    return result


@ranking_bp.route("/submit-comparison", methods=["POST"])
def submit_comparison():
    processor = _get_processor()
    _start = time.perf_counter()

    payload = request.get_json()
    if not payload:
        result = jsonify({"error": "Missing request body"}), 400
        return result

    filename_a = payload["filename_a"]
    filename_b = payload["filename_b"]
    winner = payload["winner"]
    if not all([filename_a, filename_b, winner]):
        result = jsonify({"error": "Missing required fields"}), 400

        return result
    if filename_a == filename_b:
        result = jsonify({"error": "Cannot compare image to itself"}), 400

        return result
    if winner not in [filename_a, filename_b]:
        result = jsonify({"error": "Winner must be one of the images"}), 400

        return result

    success = comparison_recorder.record_comparison(
        filename_a, filename_b, winner, 1.0, 0
    )
    if not success:
        result = jsonify({"error": "Failed to record comparison"}), 500

        return result

    data_a = state.get_cached_image(filename_a)
    data_b = state.get_cached_image(filename_b)
    if data_a is None or data_b is None:
        result = jsonify({"error": "Image not found"}), 404

        return result

    if processor:
        with processor.recent_lock:
            processor.clear_old_cache(force=False)

    result = jsonify(
        {
            "ok": True,
            "images": {
                filename_a: {
                    "score": round(float(data_a["score"]), 3),
                    "comparison_count": int(data_a["comparison_count"]),
                },
                filename_b: {
                    "score": round(float(data_b["score"]), 3),
                    "comparison_count": int(data_b["comparison_count"]),
                },
            },
        }
    )
    return result


@ranking_bp.route("/sync-all", methods=["POST"])
def sync_all_to_json():
    _start = time.perf_counter()
    images = get_all_images()
    all_comparisons = get_all_comparisons()
    count = 0
    errors = 0
    for img in images:
        success = sync_image_metadata_to_json(
            filename=img["filename"],
            score=float(img["score"]),
            rating_mu=float(img["rating_mu"]),
            rating_sigma=float(img["rating_sigma"]),
            comparison_count=int(img["comparison_count"]),
            all_comparisons=all_comparisons,
        )
        if success:
            count += 1
        else:
            errors += 1
    result = jsonify(
        {"status": "success", "synced_count": count, "error_count": errors}
    )
    return result


def register_ranking_routes(app) -> None:
    _start = time.perf_counter()
    app.register_blueprint(ranking_bp)
