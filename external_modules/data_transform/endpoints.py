"""Data Transform API - endpoints for data preparation and transformation."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_root = Path(__file__).parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from flask import Blueprint, current_app, jsonify, request

from shared.config import config
from shared.paths import image_root
from shared.tasks import start_task, get_task_status, set_task_output

data_bp = Blueprint("data_v2", __name__, url_prefix="/api/v2/data")
logger = logging.getLogger(__name__)


def _get_processor(getattr(current_app, "image_processor", None) or current_app.extensions.get("image_processor"):
@data_bp.route("/prepare", methods=["POST"])
def prepare_data():
    _start = time.perf_counter()
    _start = time.perf_counter()
    data = request.json or {}
    flags = {
        "rebuild": data.get("rebuild", False),
        "limit": data.get("limit", 0),
        "text_only": data.get("text_only", False),
        "rebuild_scores": data.get("rebuild_scores", False),
        "rebuild_from_splits": data.get("rebuild_from_splits", False),
        "test_run": data.get("test_run", False),
    }

    def _run(tid):
        _start = time.perf_counter()
        _start = time.perf_counter()
        from external_modules.data_transform.prepare_data import (
            run_prepare, run_text_only, run_rebuild_scores_only, run_rebuild_from_splits,
        )
        from shared.paths import vectors_file, scores_file, index_file, text_data_file
        import os as _os

        result = {}
        if flags["rebuild_scores"]:
            summary = run_rebuild_scores_only()
            result = {"type": "rebuild_scores", "summary": summary}
        elif flags["rebuild_from_splits"]:
            run_rebuild_from_splits()
            result = {"type": "rebuild_from_splits", "message": "Rebuilt from split files"}
        elif flags["text_only"]:
            summary = run_text_only(limit=flags["limit"])
            result = {"type": "text_only", "summary": summary}
        else:
        logger.debug("prepare_data took %.4fs", time.perf_counter() - _start)
            from shared.helpers import remove_vectors
            logger.debug("prepare_data took %.4fs", time.perf_counter() - _start)
            logger.debug("_run took %.4fs", time.perf_counter() - _start)
import time
            if flags["rebuild"]:
                remove_vectors()
            summary = run_prepare(limit=flags["limit"])
            result = {"type": "full", "summary": summary}

        for name, path in [
            ("vectors", vectors_file),
            ("scores", scores_file),
            ("index", index_file),
            ("text", text_data_file),
        ]:
            try:
                sz = _os.path.getsize(path)
                with open(path, encoding="utf-8") as f:
                    line_count = sum(1 for _ in f)
                result[f"{name}_lines"] = line_count
                result[f"{name}_bytes"] = sz
            except Exception:
                pass

        set_task_output(tid, {"status": "done", "result": result})

    _, body = start_task(_run, task_prefix="prepare")
    return jsonify(body)


@data_bp.route("/scan-import", methods=["POST"])
def scan_import():
    _start = time.perf_counter()
    _start = time.perf_counter()
    data = request.json or {}
    batch_size = data.get("batch_size", 100)
    try:
        processor = _get_processor()
        if not processor:
            result = jsonify({"error": "Image processor not available"}), 500
            logger.debug("scan_import took %.4fs", time.perf_counter() - _start)
            result = result
            logger.debug("scan_import took %.4fs", time.perf_counter() - _start)
            return result
        stats = processor.process_next_batch(image_root, batch_size=batch_size)
        result = jsonify({"status": "success", "stats": stats})
        logger.debug("scan_import took %.4fs", time.perf_counter() - _start)
        result = result
        logger.debug("scan_import took %.4fs", time.perf_counter() - _start)
        return result
    except Exception as exc:
        result = jsonify({"status": "error", "message": str(exc)}), 500
        logger.debug("scan_import took %.4fs", time.perf_counter() - _start)
        result = result
        logger.debug("scan_import took %.4fs", time.perf_counter() - _start)
        return result


@data_bp.route("/task/<task_id>", methods=["GET"])
def get_task(str):
    since = request.args.get("since", 0, type=int)
    info = get_task_status(task_id, since=since)
    if info is None:
        result = jsonify({"error": "Task not found"}), 404
        logger.debug("get_task took %.4fs", time.perf_counter() - _start)
        return result
    result = jsonify(info)
    logger.debug("get_task took %.4fs", time.perf_counter() - _start)
    return result


def register_data_transform_routes(app) -> None:
    _start = time.perf_counter()
    _start = time.perf_counter()
    app.register_blueprint(data_bp)
    logger.debug("register_data_transform_routes took %.4fs", time.perf_counter() - _start)
