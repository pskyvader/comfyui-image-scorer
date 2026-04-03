from typing import Any
import os
import random
from itertools import product, islice
import numpy as np

from shared.paths import models_dir, hyperparameters_statistics
from shared.training.model_trainer import model_trainer, grid_base, around
from shared.config import config
from shared.io import load_json, atomic_write_json

_last_used_keys: dict[str, list[str]] = {}


def load_statistics() -> dict[str, dict[str, int]]:
    data, _ = load_json(hyperparameters_statistics, dict, {})
    return data


def save_statistics(statistics: dict[str, dict[str, int]]):
    atomic_write_json(hyperparameters_statistics, statistics, indent=4)


def prepare_optimization_setup(
    base_cfg: dict[str, Any], strategy: str
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
    param_grid: dict[str, Any], max_combos: int
) -> list[dict[str, Any]]:
    keys = list(param_grid.keys())
    value_lists = [list(param_grid[k]) for k in keys]
    all_combos = [dict(zip(keys, vals)) for vals in product(*value_lists)]
    # random.shuffle(all_combos)
    return list(islice(iter(all_combos), max_combos))


def evaluate_hyperparameter_combo(
    current_cfg: dict[str, Any],
    temp_model_base: str,
    X: np.ndarray,
    y: np.ndarray,
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
    current_config: dict[str, Any],
    score: float,
    t_time: float,
    primary_metric: str,
    higher_is_better: bool,
):
    training_section = config["training"]
    top_cfg = training_section["top"]
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


def update_fastest_config(
    current_config: dict[str, Any],
    score: float,
    t_time: float,
    primary_metric: str,
    higher_is_better: bool,
):
    training_section = config["training"]
    top_cfg = training_section["top"]

    current_top_score = top_cfg["best_score"] if "best_score" in top_cfg else -1000000.0

    fast_cfg = training_section["fastest"]
    current_fast_score = (
        fast_cfg["best_score"] if "best_score" in fast_cfg else -1000000.0
    )
    current_fast_time = (
        fast_cfg["training_time"] if "training_time" in fast_cfg else 999999.0
    )

    better = (
        score > current_fast_score if higher_is_better else score < current_fast_score
    )

    better_weighted = (
        score > current_top_score * 0.95
        if higher_is_better
        else score < current_top_score * 1.05
    )

    cond_b = (
        t_time < min(current_fast_time * 0.95, current_fast_time - 1)
    ) and better_weighted

    if better or cond_b:
        print(
            f"--> Found new FASTEST model! Old {primary_metric} (fastest): {current_fast_score:.6f}, {current_fast_time:.4f} (Score: {score:.6f}, Time: {t_time:.4f}s) ( {"Higher" if higher_is_better else "Lower"} is better)"
        )
        fast_cfg.update(current_config)
        return True
    return False


def update_slowest_config(
    current_config: dict[str, Any],
    score: float,
    t_time: float,
    primary_metric: str,
    higher_is_better: bool,
):
    training_section = config["training"]

    slow_cfg = training_section["slowest"]

    current_slow_score = (
        slow_cfg["best_score"] if "best_score" in slow_cfg else -1000000.0
    )

    current_slow_time = (
        slow_cfg["training_time"] if "training_time" in slow_cfg else 1000000.0
    )

    better = (
        score > current_slow_score if higher_is_better else score < current_slow_score
    )

    better_weighted = (
        score > 0.95 * current_slow_score
        if higher_is_better
        else score < 1.05 * current_slow_score
    )

    cond_a = better and t_time < max(current_slow_time * 1.1, current_slow_time + 10)
    cond_b = (
        t_time > 60
        and better_weighted
        and t_time < min(current_slow_time * 0.99, current_slow_time - 1)
    )
    if cond_a or cond_b:
        print(
            f"--> Found new SLOWEST model! Old {primary_metric} (slowest) {current_slow_score:.6f},{current_slow_time:.4f}s (Score: {score:.6f}, Time: {t_time:.4f}s) ( {"Higher" if higher_is_better else "Lower"} is better:)"
        )
        slow_cfg.update(current_config)
        return True
    return False


def optimize_hyperparameters(
    base_cfg: dict[str, Any],
    max_combos: int,
    X: np.ndarray,
    y: np.ndarray,
    strategy: str = "generic",
) -> list[tuple[dict[str, Any], dict[str, float]]]:

    param_grid, temp_model_base = prepare_optimization_setup(base_cfg, strategy)
    # todo: use tqdm for progress bar and logging instead of print statements
    print(f"Optimizing hyperparameters over grid: {param_grid}")

    combos = generate_combos(param_grid, max_combos)
    current_cfg = base_cfg.copy()
    results: list[tuple[dict[str, Any], dict[str, float]]] = []
    new_score = False
    # for i, combo in enumerate(tqdm(combos,position=1, desc=f"HPO Search ({strategy})", unit="combo")):
    for i, combo in enumerate(combos):
        merged = {**current_cfg, **combo}
        score, t_time, primary_metric = evaluate_hyperparameter_combo(
            merged, temp_model_base, X=X, y=y
        )
        print(
            f"[{i+1}/{len(combos)}] score ({primary_metric})={score:.6f}, time={t_time:.4f}s"
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
        # Update Fastest (Track Restricted)
        if strategy == "FASTEST":
            found_new = update_fastest_config(
                merged, score, t_time, primary_metric, higher_is_better
            )
            if found_new:
                new_score = True
        # Update Slowest (Track Restricted)
        if strategy == "SLOWEST":
            found_new = update_slowest_config(
                merged, score, t_time, primary_metric, higher_is_better
            )
            if found_new:
                new_score = True
            if found_new and t_time > 300.0:
                print(
                    "Slowest improved but still too slow; skipping remaining combos in this batch."
                )
                break

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
