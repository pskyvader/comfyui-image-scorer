"""Training & Hyperparameters API - endpoints for model training and HPO."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_root = Path(__file__).parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from flask import Blueprint, jsonify, request

from shared.config import config
from shared.tasks import start_task, get_task_status, set_task_output

training_bp = Blueprint("training_v2", __name__, url_prefix="/api/v2/training")
logger = logging.getLogger(__name__)


@training_bp.route("/train", methods=["POST"])
def train_model():
    _start = time.perf_counter()
    _start = time.perf_counter()
    data = request.json or {}
    strategy = data.get("strategy", "top1")
    max_combos = data.get("max_combos", 4)

    def _run(tid):
        _start = time.perf_counter()
        _start = time.perf_counter()
        import numpy as np
        from shared.io import load_single_jsonl
        from shared.paths import vectors_file, scores_file

        retrain = data.get("retrain", False) if strategy == "hpo-cycle" else False

        if retrain:
            from shared.loaders import training_loader as training_loader_module

            logger.info("Retrain mode: removing cached models and data...")
            training_loader_module.training_loader.remove_training_models()

        X_list = load_single_jsonl(vectors_file)
        y_list = load_single_jsonl(scores_file)
        X = np.array(
            [list(v.values()) if isinstance(v, dict) else v for v in X_list],
            dtype=np.float32,
        )
        y = np.array(
            [float(s["score"]) if isinstance(s, dict) else float(s) for s in y_list],
            dtype=np.float32,
        )

        from shared.training.data_transformer import data_transformer

        n_total = X.shape[1]
        logger.info("Feature filtering: %d raw features loaded", n_total)
        try:
            training_sub = config["training"]
        except Exception:
            logger.exception("Missing training configuration")
            set_task_output(
                tid,
                {
                    "status": "error",
                    "error": "Missing training configuration (training_config.json)",
                },
            )
            logger.debug("_run took %.4fs", time.perf_counter() - _start)
            logger.debug("train_model took %.4fs", time.perf_counter() - _start)
            logger.debug("_run took %.4fs", time.perf_counter() - _start)
            logger.debug("train_model took %.4fs", time.perf_counter() - _start)
            return

        try:
            filter_steps = int(training_sub["top1"]["n_estimators"])
        except Exception:
            set_task_output(
                tid,
                {
                    "status": "error",
                    "error": "Missing 'top1' base config in training_config.json; required for training.",
                },
            )
            logger.debug("_run took %.4fs", time.perf_counter() - _start)
            logger.debug("train_model took %.4fs", time.perf_counter() - _start)
            logger.debug("_run took %.4fs", time.perf_counter() - _start)
            logger.debug("train_model took %.4fs", time.perf_counter() - _start)
            return

        X_filtered, kept_indices = data_transformer.filter_unused_features(
            X, y, steps=filter_steps
        )
        n_kept = X_filtered.shape[1]
        if n_kept < n_total:
            logger.info(
                "Feature filtering: kept %d / %d features (dropped %d)",
                n_kept,
                n_total,
                n_total - n_kept,
            )
        else:
            logger.info("Feature filtering: all %d features kept", n_total)
        X = X_filtered

        if strategy == "optimize":
            from external_modules.training_hyperparameters.run import (
                optimize_hyperparameters,
            )

            base_cfg = config["training"]["top1"]
            results = optimize_hyperparameters(
                base_cfg, max_combos, X, y, strategy="generic"
            )
            top_results = []
            for cfg, _metrics in results[:5]:
                entry = {}
                for k, v in cfg.items():
                    if isinstance(v, (int, float)):
                        entry[k] = round(float(v), 4)
                top_results.append(entry)
            from shared.loaders import training_loader as training_loader_module
            from shared.training.model_trainer import model_trainer

            best_cfg = config["training"]["top1"]
            final_model, final_metrics = model_trainer.train_model(
                config_dict=best_cfg, X=X, y=y, enable_plotting=False
            )
            training_loader_module.training_loader.save_training_model(
                final_model,
                {
                    "strategy": "optimize",
                    **{
                        k: float(v) if isinstance(v, (int, float)) else v
                        for k, v in final_metrics.items()
                    },
                },
            )
            set_task_output(
                tid,
                {
                    "status": "done",
                    "result": {
                        "type": "optimize",
                        "total_combos": len(results),
                        "top_results": top_results,
                        "strategy": strategy,
                        "model_saved": True,
                    },
                },
            )
        elif strategy == "hpo-cycle":
            from external_modules.training_hyperparameters.hyperparameter_optimizer import (
                hpo_cycle,
                reset_hyperparameters,
                evaluate_base_scores,
            )

            num_steps = data.get("num_steps", 100)
            reset = data.get("reset", False)
            evaluate_base = data.get("evaluate_base", False)
            if reset:
                reset_hyperparameters(evaluate=evaluate_base, X=X, y=y, force=True)
            elif evaluate_base:
                evaluate_base_scores(X, y)
            result = hpo_cycle(X, y, num_steps=num_steps, max_combos=max_combos)
            configs_summary = []
            for i, cfg in enumerate(result["configs"]):
                entry = {"index": i + 1}
                for k, v in cfg.items():
                    if isinstance(v, (int, float)):
                        entry[k] = round(float(v), 4)
                configs_summary.append(entry)
            set_task_output(
                tid,
                {
                    "status": "done",
                    "result": {
                        "type": "hpo_cycle",
                        "cycle": result["cycle"],
                        "configs": configs_summary,
                    },
                },
            )
        else:
            from shared.training.model_trainer import model_trainer
            from shared.loaders import training_loader as training_loader_module

            base_cfg = config["training"][strategy]
            model, metrics = model_trainer.train_model(
                config_dict=base_cfg, X=X, y=y, enable_plotting=False
            )
            metrics_dict = {
                "score": float(metrics["score"]),
                "training_time": float(metrics["training_time"]),
                "primary_metric": str(metrics["primary_metric"]),
            }
            training_loader_module.training_loader.save_training_model(
                model, {"strategy": strategy, **metrics_dict}
            )
            set_task_output(tid, {"status": "done", "result": metrics_dict})

    _, body = start_task(_run, task_prefix="train")
    return jsonify(body)


@training_bp.route("/hpo", methods=["POST"])
def run_hpo():
    _start = time.perf_counter()
    _start = time.perf_counter()
    data = request.json or {}
    reset = data.get("reset", False)
    cycles = data.get("cycles", None)
    num_steps = data.get("num_steps", None)
    max_combos = data.get("max_combos", None)

    def _run(tid):
        _start = time.perf_counter()
        _start = time.perf_counter()
        from external_modules.training_hyperparameters.hyperparameter_optimizer import (
            run_hpo_cycles,
        )

        res = run_hpo_cycles(
            reset=reset, cycles=cycles, num_steps=num_steps, max_combos=max_combos
        )
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
        logger.debug("_run took %.4fs", time.perf_counter() - _start)

 logger.debug("run_hpo took %.4fs", time.perf_counter() - _start)
    _, body = start_task(_run, task_prefix="hpo")
    logger.debug("run_hpo took %.4fs", time.perf_counter() - _start)
    return jsonify(body)


@training_bp.route("/remove-models", methods=["POST"])
def delete_models():
    _start = time.perf_counter()
    _start = time.perf_counter()
    try:
        from shared.helpers import remove_models
        logger.debug("delete_models took %.4fs", time.perf_counter() - _start)
import time

        remove_models()
        return jsonify({"status": "success"})
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@training_bp.route("/task/<task_id>", methods=["GET"])
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


@training_bp.route("/config", methods=["GET"])
def get_training_config():
    _start = time.perf_counter()
    _start = time.perf_counter()
    try:
        training = config["training"]
    except KeyError:
        result = jsonify({"error": "Training config not found"}), 404
        logger.debug("get_training_config took %.4fs", time.perf_counter() - _start)
        result = result
        logger.debug("get_training_config took %.4fs", time.perf_counter() - _start)
        return result
    try:
        data = training.copy()
    except Exception:
        data = dict(training)
    result = jsonify({"status": "ok", "config": data})
    logger.debug("get_training_config took %.4fs", time.perf_counter() - _start)
    result = result
    logger.debug("get_training_config took %.4fs", time.perf_counter() - _start)
    return result


@training_bp.route("/config", methods=["POST"])
def update_training_config():
    _start = time.perf_counter()
    _start = time.perf_counter()
    data = request.json or {}
    overwrite = bool(data.get("overwrite", False))
    payload = data.get("config") if "config" in data else data
    if not isinstance(payload, dict):
        result = jsonify({"error": "Invalid config payload"}), 400
        logger.debug("update_training_config took %.4fs", time.perf_counter() - _start)
        result = result
        logger.debug("update_training_config took %.4fs", time.perf_counter() - _start)
        return result
    try:
        if overwrite:
            config["training"] = payload
        else:
            training = config["training"]
            for k, v in payload.items():
                training[k] = v
        result = jsonify({"status": "ok", "config": config["training"].copy()})
        logger.debug("update_training_config took %.4fs", time.perf_counter() - _start)
        result = result
        logger.debug("update_training_config took %.4fs", time.perf_counter() - _start)
        return result
    except Exception as e:
        logger.exception("Failed to update training config")
        result = jsonify({"status": "error", "message": str(e)}), 500
        logger.debug("update_training_config took %.4fs", time.perf_counter() - _start)
        result = result
        logger.debug("update_training_config took %.4fs", time.perf_counter() - _start)
        return result


def register_training_routes(app) -> None:
    _start = time.perf_counter()
    _start = time.perf_counter()
    app.register_blueprint(training_bp)
    logger.debug("register_training_routes took %.4fs", time.perf_counter() - _start)
