import os
import random
from itertools import product, islice
from typing import Any

import numpy as np
from numpy import typing as npt

from ...shared.paths import models_dir
from ...shared.training.model_trainer import grid_base, around, model_trainer
from ...shared.training.data_transformer import data_transformer
from ...shared.loaders.training_loader import training_loader
from ...shared.config import config
from .config_utils import (
    generate_random_config,
    generate_fastest_setup,
    generate_slowest_setup,
    crossover_config,
)
from .run import evaluate_hyperparameter_combo

import time

from ...shared.logger import get_logger

logger = get_logger(__name__)

# STATE_FILE = os.path.join(models_dir, "hyperparameter_state.json")
NUM_CONFIGS = 5

# Guard to prevent re-entrant or concurrent HPO loop runs from within the
# notebook or other callers. The HPO loop must be started explicitly and
# may not be invoked indirectly more than once at a time.
_hpo_running = False


def _mapping_get(mapping: Any, key: str, default: Any = None) -> Any:
    if key in mapping:
        return mapping[key]
    return default





def _load_state() -> dict[str, Any]:
    training_config = config["training"]
    data = []
    data.append(training_config["top1"])
    data.append(training_config["top2"])
    data.append(training_config["top3"])
    data.append(training_config["top4"])
    data.append(training_config["top5"])

    result = {
        "configs": data,
        "step": 0,
        "cycle": 0,
        "used_keys": (
            training_config["used_keys"] if "used_keys" in training_config else []
        ),
    }
    return result

    # data, err = load_json(STATE_FILE, dict, None)
    # return data if data is not None else {}


def _save_state(state: dict[str, Any]):
    configs = state["configs"]
    config["training"]["top1"] = configs[0]
    config["training"]["top2"] = configs[1]
    config["training"]["top3"] = configs[2]
    config["training"]["top4"] = configs[3]
    config["training"]["top5"] = configs[4]

    config["training"]["used_keys"] = state["used_keys"]

    # atomic_write_json(STATE_FILE, state, indent=4)


def list_filtered_features(kept_indices: npt.NDArray[np.float32]):

    features: dict[str, dict[str, list[int]]] = {
        name: {"kept": [], "removed": []}
        for name in data_transformer.vector_ranges.keys()
    }
    for position, feature in data_transformer.feature_to_vector.items():
        name = feature["vector_name"]
        local_position = feature["position_in_vector"]

        if position in kept_indices:
            features[name]["kept"].append(local_position)
        else:
            features[name]["removed"].append(local_position)

    for name, (feature) in features.items():
        kept: list[int] = feature["kept"]
        removed: list[int] = feature["removed"]
        slot_size = data_transformer.vector_ranges[name]["slot_size"]
        logger.debug(
            f" {name}: \n"
            f"kept: {len(kept)}/{slot_size} ({(100*len(kept)/slot_size):.1f}%)"
            # f"removed: {len(removed)}/{slot_size} ({(100*len(removed)/slot_size):.1f}%)"
        )


def load_training_data(
    retrain: bool = False, filter_comparisons: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    if retrain:
        logger.info("Retrain mode: removing cached models and data...")
        training_loader.remove_training_models()
    logger.info("loading raw data ...")
    x, y = data_transformer.get_raw_data()
    logger.info(f"raw data loaded, shape: {x.shape}, {y.shape}")

    filter_steps = 1000
    logger.info(f"filtering unused features after {filter_steps} steps...")
    x, kept_indices = data_transformer.filter_unused_features(x, y, steps=filter_steps)
    list_filtered_features(kept_indices)
    if filter_comparisons:
        logger.info("filtering low comparison data ...")
        threshold = config["training"]["min_comparisons_threshold"]
        x, y = data_transformer.filter_low_comparisons(x, y, threshold=threshold)
        logger.info(
            f"filtered data by count (>={threshold}), shape: {x.shape}, {y.shape}"
        )

    logger.info(f"Feature filtering complete: X={x.shape}")
    result = (x, y)
    return result


def _evaluate_config(
    cfg: dict[str, Any], X: np.ndarray, y: np.ndarray
) -> dict[str, Any]:
    os.makedirs(models_dir, exist_ok=True)
    temp_model_base = os.path.join(models_dir, "temp_model")
    score, t_time, primary_metric = evaluate_hyperparameter_combo(
        cfg, temp_model_base, X=X, y=y
    )
    return {
        **cfg,
        "best_score": score,
        "training_time": t_time,
        "primary_metric": primary_metric,
    }


def reset_hyperparameters():
    top1 = generate_random_config()
    top2 = generate_random_config()
    top3 = generate_slowest_setup()
    top4 = generate_fastest_setup()
    top5 = generate_random_config()
    configs = [top1, top2, top3, top4, top5]
    state = {"configs": configs, "step": 0, "cycle": 0, "used_keys": []}

    _save_state(state)

    return state


def evaluate_base_scores(x: np.ndarray, y: np.ndarray):
    state = _load_state()
    if (
        not state
        or "configs" not in state
        or len(state.get("configs", [])) != NUM_CONFIGS
    ):
        raise RuntimeError(
            "HPO state missing or invalid. To create base configs, call reset_hyperparameters(force=True) directly."
        )
    logger.info("Evaluating base scores for current configs...")
    for i in range(NUM_CONFIGS):
        logger.info(f"  Config {i + 1}: evaluating...")
        state["configs"][i] = _evaluate_config(state["configs"][i], X, y)
        logger.info(
            f"    score={_mapping_get(state['configs'][i], 'best_score', -1):.6f}  time={_mapping_get(state['configs'][i], 'training_time', 0):.2f}s"
        )
    _save_state(state)
    logger.info("Base scores updated.")
    result = state
    return result


def get_hpo_state() -> dict[str, Any]:
    _start = time.perf_counter()
    _start = time.perf_counter()
    state = _load_state()
    if (
        not state
        or "configs" not in state
        or len(state.get("configs", [])) != NUM_CONFIGS
    ):
        raise RuntimeError(
            "HPO state missing or invalid. To initialize base configs, call reset_hyperparameters(force=True) explicitly."
        )
    result = state
    return result


def _run_step_on_config(
    cfg: dict[str, Any],
    used_keys: list[str],
    X: np.ndarray,
    y: np.ndarray,
    max_combos: int,
) -> tuple[dict[str, Any], list[str]]:
    all_keys = list(grid_base.keys())
    random.shuffle(all_keys)

    chosen_key = None
    for key in all_keys:
        if key not in used_keys:
            chosen_key = key
            break
    if chosen_key is None:
        chosen_key = all_keys[0]
        used_keys = []

    varied_vals = around(chosen_key, cfg[chosen_key])
    logger.info(
        f"    Varying: {chosen_key} (current={cfg[chosen_key]:.6g}) -> {[round(v, 6) for v in varied_vals]}"
    )
    logger.info(f"    Recently used params: {used_keys}")

    param_grid = {k: [cfg[k]] for k in all_keys}
    param_grid[chosen_key] = varied_vals

    keys_grid = list(param_grid.keys())
    value_lists = [list(param_grid[k]) for k in keys_grid]
    all_combos = [dict(zip(keys_grid, vals)) for vals in product(*value_lists)]
    combos = list(islice(iter(all_combos), max_combos))

    os.makedirs(models_dir, exist_ok=True)
    temp_model_base = os.path.join(models_dir, "temp_model")

    best_cfg = cfg.copy()
    best_score = cfg.get("best_score")
    best_time = cfg.get("training_time")
    training_objective = config["training"]["objective"]
    improved = False

    for i, combo in enumerate(combos):
        logger.info("---" * 10)
        merged = {**cfg, **combo}
        score, t_time, primary_metric = evaluate_hyperparameter_combo(
            merged, temp_model_base, X=X, y=y
        )
        directions = model_trainer.METRIC_DIRECTIONS.get(training_objective, {})
        higher_is_better = directions.get(primary_metric, True)

        arrow = ""
        if (higher_is_better and score > best_score) or (
            not higher_is_better and score < best_score
        ):
            best_score = score
            best_time = t_time
            best_cfg = {**merged, "best_score": score, "training_time": t_time}
            arrow = "  <-- NEW BEST"
            improved = True

        logger.info(
            f"      Combo {i+1}/{len(combos)}: {chosen_key}={combo[chosen_key]:.6g}  score={score:.6f}  time={t_time:.2f}s{arrow}"
        )

    if improved:
        logger.info(f"    Config improved: score={best_score:.6f}, time={best_time:.2f}s")
    else:
        logger.info(f"    Config unchanged: best score remains {best_score:.6f}")
        used_keys.append(chosen_key)

    if len(used_keys) > len(all_keys):
        used_keys = used_keys[-len(all_keys) // 3 :]

    return best_cfg, used_keys


def hpo_cycle(
    X: np.ndarray,
    y: np.ndarray,
    num_steps: int = 100,
    max_combos: int = 4,
    cycle: int = 0,
) -> dict[str, Any]:
    global _hpo_running
    if _hpo_running:
        raise RuntimeError(
            "HPO loop is already running. Concurrent or nested runs are not allowed."
        )
    _hpo_running = True
    try:
        state = _load_state()
        if (
            not state
            or "configs" not in state
            or len(state.get("configs", [])) != NUM_CONFIGS
        ):
            raise RuntimeError(
                "HPO state missing or invalid. The HPO loop will not auto-create base configs. Call reset_hyperparameters(force=True) to initialize."
            )

        # Require the canonical training configuration to include top1..top4.
        try:
            training_sub = config["training"]
        except Exception:
            raise RuntimeError(
                "Missing training configuration (training_config.json). Initialize base configs with reset_hyperparameters(force=True)."
            )

        if any(k not in training_sub for k in ("top1", "top2", "top3", "top4", "top5")):
            raise RuntimeError(
                "training_config.json missing base entries ('top1'..'top4','top5'). The HPO loop will not create them; call reset_hyperparameters(force=True) to initialize."
            )

        configs = state["configs"]
        used_keys = state.get("used_keys", [])
        step_start = state.get("step", 0)

        logger.info(f"\n{'='*80}")
        logger.info(f"HPO Cycle {cycle + 1} — Starting from step {step_start}/{num_steps}")

        for i in range(step_start, num_steps):
            idx = i % NUM_CONFIGS
            logger.info(f"\n{'---' * 25}")
            logger.info(f"Step {i + 1}/{num_steps}  —  Config {idx + 1}")
            cfg = configs[idx]
            logger.info(
                f"best_score={cfg['best_score']:.6f}  training_time={cfg.get('training_time'):.2f}s"
            )
            logger.info(f" {cfg}")
            # for key in grid_base.keys():
            #     logger.info(f"  {key}={cfg.get(key):.6g}")

            configs[idx], used_keys = _run_step_on_config(
                configs[idx], used_keys, X, y, max_combos
            )
            state["step"] = i + 1
            state["configs"] = configs
            state["used_keys"] = used_keys
            _save_state(state)

        logger.info(f"\n{'='*80}")
        logger.info(f"Cycle complete — Sorting configs by score")
        # Determine sort direction from model_trainer metric preferences so that
        # 'best' is consistent with the training objective (higher_is_better).
        try:
            training_objective = config["training"].get("objective")
        except Exception:
            training_objective = None

        directions = getattr(model_trainer, "METRIC_DIRECTIONS", {}).get(
            training_objective, {}
        )
        if directions:
            # If any configured metric expects higher-is-better, treat sorting
            # as higher-is-better; otherwise assume lower-is-better.
            higher_is_better = any(bool(v) for v in directions.values())
        else:
            higher_is_better = True

        logger.info(
            f"Sorting configs with higher_is_better={higher_is_better} (objective={training_objective})"
        )
        configs.sort(
            key=lambda c: _mapping_get(c, "best_score", -1000000.0),
            reverse=higher_is_better,
        )
        for i, c in enumerate(configs):
            logger.info(
                f"  Rank {i + 1}: score={_mapping_get(c, 'best_score', -1):.6f}  time={_mapping_get(c, 'training_time', 0):.2f}s"
            )

        logger.info(
            f"\nBreeding next generation — keeping top 2, creating 2 children via crossover"
        )
        parents = [configs[0], configs[1]]
        child1 = crossover_config(dict(parents[0]), dict(parents[1]))
        child2 = crossover_config(dict(parents[0]), dict(parents[1]))
        random_child = generate_random_config()
        logger.info(f"  Parent 1:  score={_mapping_get(parents[0], 'best_score', -1):.6f}")
        logger.info(f"  Parent 2:  score={_mapping_get(parents[1], 'best_score', -1):.6f}")
        logger.info(f"  Child 1:   score=NEW (to be evaluated next cycle)")
        logger.info(f"  Child 2:   score=NEW (to be evaluated next cycle)")
        logger.info(f"  random child:   score=NEW (to be evaluated next cycle)")

        new_configs = [parents[0], parents[1], child1, child2, random_child]
        new_state = {
            "configs": new_configs,
            "step": 0,
            "cycle": cycle + 1,
            "used_keys": used_keys,
        }
        _save_state(new_state)

        logger.info(f"\nCycle {cycle + 1} complete. Trigger again to start next cycle.")
        logger.info(f"{'='*80}\n")
        return new_state
    finally:
        _hpo_running = False


def run_hpo_cycles(
    cycles: int,
    num_steps: int,
    max_combos: int,
):
    """
    Helper to run multiple HPO cycles for API/frontend use.

    - If `reset` is True, calls `reset_hyperparameters(force=True)` to create base configs.
    - Otherwise requires existing HPO state and `top1`..`top4` in `training_config.json`.
    Returns a list of `hpo_cycle` results (each is the new_state dict).
    """

    # load_training_data will validate presence of top1..top4 and raise if missing
    X, y = load_training_data(retrain=False)

    results = []
    for i in range(cycles):
        logger.info(f"[run_hpo_cycles] Starting cycle {i+1}/{cycles}")
        res = hpo_cycle(X, y, num_steps=num_steps, max_combos=max_combos)
        results.append(res)
    return results
