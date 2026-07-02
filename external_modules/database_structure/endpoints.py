"""Database endpoints - API routes for maintenance and file operations."""

from flask import Blueprint, jsonify, request, current_app

from .images_table import get_all_images, reset_all_image_ratings, get_image_count
from .comparisons_table import (
    get_all_comparisons,
    get_total_comparisons,
    clean_comparisons,
)
from .cleanup_orphans import cleanup_orphans
from .deduplicate_scored import deduplicate_scored
from .path_handler import sync_image_metadata_to_json

from pathlib import Path
import time

from shared.paths import image_root_processed

from ...shared.graph.crystal_graph import crystal_graph
from ...shared.tasks import start_task, get_task_status, set_task_output

from ...shared.logger import get_logger, ModuleLogger

logger: ModuleLogger = get_logger(__name__)
database_bp = Blueprint("database", __name__, url_prefix="/api/v2/database")


def _get_processor():
    """Get the image_processor instance from the Flask app."""
    return getattr(current_app, "image_processor", None) or current_app.extensions.get(
        "image_processor"
    )


@database_bp.route("/status", methods=["GET"])
def get_status():
    result = jsonify(
        {
            "status": "ok",
            "images": get_image_count(),
            "comparisons": get_total_comparisons(),
        }
    )
    return result


@database_bp.route("/normalize-comparisons", methods=["POST"])
def normalize():
    _start = time.perf_counter()
    _start = time.perf_counter()
    try:
        stats = clean_comparisons()
        crystal_graph.rebuild_from_database()
        result = jsonify({"status": "success", "stats": stats})

        return result
    except Exception as exc:
        result = jsonify({"status": "error", "message": str(exc)}), 500
        return result


@database_bp.route("/rebuild-db", methods=["POST"])
def rebuild_database():
    _start = time.perf_counter()
    _start = time.perf_counter()
    processor = _get_processor()
    if processor is None:

        return (
            jsonify({"status": "error", "error": "Image processor not available"}),
            500,
        )

    def _run(tid):
        _start = time.perf_counter()
        _start = time.perf_counter()
        try:
            result = processor.rebuild_database_from_ranked()
            crystal_graph.rebuild_from_database()
            set_task_output(
                tid,
                {
                    "status": "done",
                    "result": result,
                },
            )
        except Exception as exc:
            logger.exception("rebuild_database failed")
            set_task_output(
                tid,
                {
                    "status": "error",
                    "error": str(exc),
                },
            )

    _, body = start_task(_run, task_prefix="rebuild", args=())
    return jsonify(body)


@database_bp.route("/sync-all", methods=["POST"])
def sync_all():
    _start = time.perf_counter()
    _start = time.perf_counter()

    def _run(tid):
        _start = time.perf_counter()
        _start = time.perf_counter()
        images = get_all_images()
        all_comparisons = get_all_comparisons()
        count = 0
        errors = 0
        total = len(images)
        print(f"Syncing {total} images...")
        for img in images:
            ok = sync_image_metadata_to_json(
                filename=img["filename"],
                score=float(img["score"]),
                rating_mu=float(img["rating_mu"]),
                rating_sigma=float(img["rating_sigma"]),
                comparison_count=int(img["comparison_count"]),
                all_comparisons=all_comparisons,
            )
            if ok:
                count += 1
            else:
                errors += 1
            print(
                f"[{count + errors}/{total}] {img['filename']} {'OK' if ok else 'FAIL'}"
            )
        print(f"Done: {count} synced, {errors} errors")
        set_task_output(
            tid,
            {
                "status": "done",
                "result": {"synced_count": count, "error_count": errors},
            },
        )

    _, body = start_task(_run, task_prefix="sync", args=())

    return jsonify(body)


@database_bp.route("/cleanup-orphans", methods=["POST"])
def run_cleanup_orphans():
    _start = time.perf_counter()
    _start = time.perf_counter()
    data = request.json or {}
    dry_run = data.get("dry_run", True)
    try:
        result = cleanup_orphans(root=None, dry_run=dry_run, delete_enabled=not dry_run)
        result = jsonify({"status": "success", "result": result})

        result = result

        return result
    except Exception as exc:
        result = jsonify({"status": "error", "message": str(exc)}), 500

        result = result

        return result


@database_bp.route("/deduplicate", methods=["POST"])
def run_deduplicate():
    _start = time.perf_counter()
    _start = time.perf_counter()

    data = request.json or {}
    dry_run = data.get("dry_run", True)
    limit = data.get("limit", 0)
    try:
        result = deduplicate_scored(
            root=Path(image_root_processed), dry_run=dry_run, limit=limit
        )
        result = jsonify({"status": "success", "result": result})

        result = result

        return result
    except Exception as exc:
        result = jsonify({"status": "error", "message": str(exc)}), 500

        result = result

        return result


@database_bp.route("/task/<task_id>", methods=["GET"])
def get_task(task_id: str):
    since = request.args.get("since", 0, type=int)
    info = get_task_status(task_id, since=since)
    if info is None:
        result = jsonify({"error": "Task not found"}), 404

        return result
    result = jsonify(info)

    return result


def register_database_routes(app):
    _start = time.perf_counter()
    _start = time.perf_counter()
    app.register_blueprint(database_bp)
