import logging
from typing import Any
import os
import random
from itertools import product, islice
import numpy as np

from shared.paths import models_dir, hyperparameters_statistics
from shared.training.model_trainer import model_trainer, grid_base, around
from shared.config import config
from shared.io import load_json, atomic_write_json
import time

logger = logging.getLogger(__name__)

_last_used_keys: dict[str, list[str]] = {}


def load_statistics() -> dict[str, dict[str, int]]:
    _start = time.perf_counter()
    _start = time.perf_counter()
    data, _ = load_json(hyperparameters_statistics, dict, {})
    result = data
    logger.debug("load_statistics took %.4fs", time.perf_counter() - _start)
    result = result
    logger.debug("load_statistics took %.4fs", time.perf_counter() - _start)
    return result


def save_statistics(dict[str, dict[str, int]]):
    atomic_write_json(hyperparameters_statistics, statistics, indent=4)


def prepare_optimization_setup(
    _start = time.perf_counter()
    _start = time.perf_counter()
    base_cfg: dict[str, Any], strategy: str
    logger.debug("prepare_optimization_setup took %.4fs", time.perf_counter() - _start)
) -> tuple[dict[str, Any], str]:
    global _last_used_keys
    current_last_used = _last_used_keys[strategy] if strategy in _last_used_keys else []
    param_grid: dict[str, Any] = {}
    chosen = 0
    limit = 1
    keys = list(grid_base.keys())
    random.shuffle(keys)
    print(f"last used keys for {strategy}: {current_last_used}")
    for key in keys:
        if key in current_last_used:
            continue
        if key not in base_cfg:
            raise ValueError(
                f"Base config missing required key '{key}' for optimization."
            )
        if chosen < limit:
            chosen += 1
            param_grid[key] = around(key, base_cfg[key])
            current_last_used.append(key)
        else:
            param_grid[key] = [base_cfg[key]]

    for key in current_last_used[:-1]:
        param_grid[key] = [base_cfg[key]]

    if len(current_last_used) == len(keys):
        current_last_used = []
    _last_used_keys[strategy] = current_last_used
    os.makedirs(models_dir, exist_ok=True)
    temp_model_base = os.path.join(models_dir, "temp_model")
    return (param_grid, temp_model_base)


def generate_combos(
    _start = time.perf_counter()
    _start = time.perf_counter()
    param_grid: dict[str, Any], max_combos: int
    logger.debug("generate_combos took %.4fs", time.perf_counter() - _start)
) -> list[dict[str, Any]]:
    keys = list(param_grid.keys())
    value_lists = [list(param_grid[k]) for k in keys]
    all_combos = [dict(zip(keys, vals)) for vals in product(*value_lists)]
    # random.shuffle(all_combos)
    return list(islice(iter(all_combos), max_combos))


def evaluate_hyperparameter_combo(
    _start = time.perf_counter()
    _start = time.perf_counter()
    current_cfg: dict[str, Any],
    temp_model_base: str,
    X: np.ndarray,
    y: np.ndarray,
    logger.debug("evaluate_hyperparameter_combo took %.4fs", time.perf_counter() - _start)
) -> tuple[float, float, str]:
    _, metrics = model_trainer.train_model(
        config_dict=current_cfg, X=X, y=y, enable_plotting=True
    )
    score = float(metrics["score"])
    training_time = float(metrics["training_time"])
    primary_metric = str(metrics["primary_metric"])
    try:
        os.remove(temp_model_base + ".npz")
    except Exception:
        pass
    try:
        os.remove(temp_model_base + ".pkl")
    except Exception:
        pass
    return score, training_time, primary_metric


def update_top_config(
    _start = time.perf_counter()
    _start = time.perf_counter()
    current_config: dict[str, Any],
    score: float,
    t_time: float,
    primary_metric: str,
    higher_is_better: bool,
    logger.debug("update_top_config took %.4fs", time.perf_counter() - _start)
):
    training_section = config["training"]
    top_cfg = training_section["top1"]
    current_top_score = top_cfg["best_score"] if "best_score" in top_cfg else -1000000.0
    better = (
        score > current_top_score if higher_is_better else score < current_top_score
    )
    if better:
        print(
            f"--> Found new TOP model! (old {primary_metric}: {current_top_score:.6f}, new: {score:.6f}) (time: {t_time:.4f}s) ( {"Higher" if higher_is_better else "Lower"} is better)"
        )
        top_cfg.update(current_config)
        return True
    return False


def optimize_hyperparameters(
    _start = time.perf_counter()
    _start = time.perf_counter()
    base_cfg: dict[str, Any],
    max_combos: int,
    X: np.ndarray,
    y: np.ndarray,
    strategy: str = "generic",
    logger.debug("optimize_hyperparameters took %.4fs", time.perf_counter() - _start)
) -> list[tuple[dict[str, Any], dict[str, float]]]:

    param_grid, temp_model_base = prepare_optimization_setup(base_cfg, strategy)
    combos = generate_combos(param_grid, max_combos)
    total = len(combos)
    logger.info("HPO grid: %d combos to evaluate", total)
    for key, vals in param_grid.items():
        logger.info("  %s: %s", key, vals)

    current_cfg = base_cfg.copy()
    results: list[tuple[dict[str, Any], dict[str, float]]] = []
    new_score = False
    best_score_so_far = float("-inf")
    for i, combo in enumerate(combos):
        merged = {**current_cfg, **combo}
        score, t_time, primary_metric = evaluate_hyperparameter_combo(
            merged, temp_model_base, X=X, y=y
        )
        if score > best_score_so_far:
            best_score_so_far = score
        logger.info(
            "[%d/%d] %s score=%.6f time=%.2fs best=%.6f",
            i + 1,
            total,
            primary_metric,
            score,
            t_time,
            best_score_so_far,
        )

        res_metrics = {"best_score": score, "training_time": t_time}
        results.append((merged, res_metrics))
        merged.update(res_metrics)
        higher_is_better = model_trainer.METRIC_DIRECTIONS[
            config["training"]["objective"]
        ][primary_metric]
        found_new = update_top_config(
            merged, score, t_time, primary_metric, higher_is_better
        )
        if found_new:
            new_score = True
    logger.info(
        "HPO complete — evaluated %d/%d combos, best score=%.6f",
        i + 1,
        total,
        best_score_so_far,
    )

    global _last_used_keys
    current_last_used = _last_used_keys[strategy] if strategy in _last_used_keys else []

    # remove last element
    if len(current_last_used) > 0:
        if new_score:
            last_element = current_last_used.pop()
        else:
            last_element = current_last_used[-1]

        statistics = load_statistics()
        if not strategy in statistics:
            statistics[strategy] = {}
        if not last_element in statistics[strategy]:
            statistics[strategy][last_element] = 0
        statistics[strategy][last_element] += 1 if new_score else -1
        save_statistics(statistics)

    return results
