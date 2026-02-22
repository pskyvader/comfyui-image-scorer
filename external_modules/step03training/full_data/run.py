from typing import Any, Dict, Tuple, List
import os
import random
from itertools import product, islice
import numpy as np

from external_modules.step03training.full_data.config_utils import grid_base, around

from shared.paths import models_dir
from shared.training.model_trainer import model_trainer
from shared.config import config

_last_used_keys: Dict[str, List[str]] = {}

def prepare_optimization_setup(
    base_cfg: Dict[str, Any], strategy: str
) -> Tuple[Dict[str, Any], str]:
    global _last_used_keys
    current_last_used = _last_used_keys[strategy] if strategy in _last_used_keys else []
    param_grid: Dict[str, Any] = {}
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
    param_grid: Dict[str, Any], max_combos: int
) -> List[Dict[str, Any]]:
    keys = list(param_grid.keys())
    value_lists = [list(param_grid[k]) for k in keys]
    all_combos = [dict(zip(keys, vals)) for vals in product(*value_lists)]
    # random.shuffle(all_combos)
    return list(islice(iter(all_combos), max_combos))


def evaluate_hyperparameter_combo(
    current_cfg: Dict[str, Any],
    temp_model_base: str,
    X: np.ndarray,
    y: np.ndarray,
) -> Tuple[float, float]:
    _, metrics = model_trainer.train_model(
        config_dict=current_cfg,
        X=X,
        y=y,
    )
    score = float(metrics["r2"])
    training_time = float(metrics["training_time"])
    try:
        os.remove(temp_model_base + ".npz")
    except Exception:
        pass
    try:
        os.remove(temp_model_base + ".pkl")
    except Exception:
        pass
    return score, training_time


def update_top_config(current_config: Dict[str, Any], score: float, t_time: float):
    training_section = config["training"]
    top_cfg = training_section["top"]
    current_top_score = top_cfg["best_score"] if "best_score" in top_cfg else -1000000.0
    if score > current_top_score:
        print(
            f"--> Found new TOP model! (old: {current_top_score:.4f}, new: {score:.4f})"
        )
        new_top: Dict[str, Any] = {
            **current_config,
            "best_score": score,
            "training_time": t_time,
        }
        top_cfg.update(new_top)
        return True
    return False


def update_fastest_config(current_config: Dict[str, Any], score: float, t_time: float):
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

    cond_a = score > current_fast_score

    cond_b = (t_time < current_fast_time*0.95) and (score >= 0.95 * current_top_score)

    if cond_a or cond_b:
        print(
            f"--> Found new FASTEST model! Old (fastest): {current_fast_score:.4f}, {current_fast_time:.4f} (Score: {score:.4f}, Time: {t_time:.4f}s)"
        )
        new_fast: Dict[str, Any] = {
            **current_config,
            "best_score": score,
            "training_time": t_time,
        }
        fast_cfg.update(new_fast)
        return True
    return False


def update_slowest_config(current_config: Dict[str, Any], score: float, t_time: float):
    training_section = config["training"]

    slow_cfg = training_section["slowest"]

    current_slow_score = (
        slow_cfg["best_score"] if "best_score" in slow_cfg else -1000000.0
    )
    
    current_slow_time = (
        slow_cfg["training_time"] if "training_time" in slow_cfg else 1000000.0
    )
    
    
    cond_a = score > current_slow_score and t_time < current_slow_time * 1.1
    cond_b = (
        t_time > 60
        and score > current_slow_score * 0.95
        and t_time
        < min(current_slow_time * 0.99, current_slow_time - 1)
    )
    if cond_a or cond_b:
        print(
            f"--> Found new SLOWEST model! Old (slowest) {current_slow_score:.4f},{current_slow_time:.4f}s (Score: {score:.4f}, Time: {t_time:.4f}s)"
        )
        new_slow: Dict[str, Any] = {
            **current_config,
            "best_score": score,
            "training_time": t_time,
        }
        slow_cfg.update(new_slow)
        return True
    return False


def optimize_hyperparameters(
    base_cfg: Dict[str, Any],
    max_combos: int,
    X: np.ndarray,
    y: np.ndarray,
    strategy: str = "generic",
) -> List[Tuple[Dict[str, Any], Dict[str, float]]]:

    param_grid, temp_model_base = prepare_optimization_setup(base_cfg, strategy)
    print(f"Optimizing hyperparameters over grid: {param_grid}")

    combos = generate_combos(param_grid, max_combos)
    current_cfg = base_cfg.copy()
    results: List[Tuple[Dict[str, Any], Dict[str, float]]] = []

    for i in range(len(combos)):
        combo = combos[i]
        print("_" * 30)
        print(
            f"Evaluating hyperparameter combo {i+1}/{len(combos)}, with params: {combo}"
        )
        merged = {**current_cfg, **combo}
        # print("merged:", merged)
        score, t_time = evaluate_hyperparameter_combo(merged, temp_model_base, X=X, y=y)
        print(f"r2={score:.4f}, time={t_time:.4f}s for Evaluated params {combo}")

        res_metrics = {"r2": score, "training_time": t_time}
        results.append((merged, res_metrics))
        update_top_config(merged, score, t_time)

        # Update Fastest (Track Restricted)
        if strategy == "FASTEST":
            update_fastest_config(merged, score, t_time)
        # Update Slowest (Track Restricted)
        if strategy == "SLOWEST":
            updated = update_slowest_config(merged, score, t_time)
            if updated and t_time > 300.0:
                print(
                    "Slowest improved but still too slow; skipping remaining combos in this batch."
                )
                break

    return results
