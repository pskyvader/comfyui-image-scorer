"""Training & Hyperparameters API - endpoints for model training and HPO."""

from __future__ import annotations

import sys
import time
from pathlib import Path

_root = Path(__file__).parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from flask import Blueprint, jsonify, request

from .hyperparameter_optimizer import (
    reset_hyperparameters,
    run_hpo_cycles,
)
from ...shared.config import config
from ...shared.helpers import remove_models
from ...shared.tasks import get_task_status, set_task_output, start_task

training_bp = Blueprint("training_v2", __name__, url_prefix="/api/v2/training")
from ...shared.logger import get_logger, ModuleLogger
logger: ModuleLogger = get_logger(__name__)


@training_bp.route("/reset", methods=["POST"])
def reset_configs():
    reset_hyperparameters()
    result = jsonify({"status": "success"})
    return result


@training_bp.route("/hpo", methods=["POST"])
def run_hpo():
    _start = time.perf_counter()
    # data = request.json
    cycles = config["training"]["cycles"]
    num_steps = config["training"]["steps_per_cycle"]
    max_combos = config["training"]["max_combos"]

    def _run(tid):
        _start = time.perf_counter()
        res = run_hpo_cycles(cycles=cycles, num_steps=num_steps, max_combos=max_combos)
        set_task_output(
            tid,
            {
                "status": "done",
                "result": {
                    "cycles_run": len(res),
                    "last_result": res[-1] if res else None,
                },
            },
        )

    _, body = start_task(_run, task_prefix="hpo", args=())

    return jsonify(body)


@training_bp.route("/remove-models", methods=["POST"])
def delete_models():
    _start = time.perf_counter()
    remove_models()
    result = jsonify({"status": "success"})

    return result


@training_bp.route("/task/<task_id>", methods=["GET"])
def get_task(task_id):
    _start = time.perf_counter()
    since = request.args.get("since", 0, type=int)
    info = get_task_status(task_id, since=since)
    if info is None:
        result = jsonify({"error": "Task not found"}), 404
    else:
        result = jsonify(info)

    return result


@training_bp.route("/config", methods=["GET"])
def get_training_config():
    _start = time.perf_counter()
    training = config["training"]
    data = training.copy()
    result = jsonify({"status": "ok", "config": data})

    return result


@training_bp.route("/config", methods=["POST"])
def update_training_config():
    _start = time.perf_counter()
    data = request.json or {}
    overwrite = bool(data.get("overwrite", False))
    payload = data["config"] if "config" in data else data
    if not isinstance(payload, dict):
        result = jsonify({"error": "Invalid config payload"}), 400

        return result
    if overwrite:
        config["training"] = payload
    else:
        training = config["training"]
        for k, v in payload.items():
            training[k] = v
    result = jsonify({"status": "ok", "config": config["training"].copy()})

    return result


def register_training_routes(app) -> None:
    _start = time.perf_counter()
    app.register_blueprint(training_bp)
