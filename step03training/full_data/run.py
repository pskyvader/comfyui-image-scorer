from typing import Any, Dict, Tuple, cast, List, Optional
import os
import time
import warnings
import random
from itertools import product, islice

import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import PowerTransformer

from shared.config import config

import lightgbm as lgb
from step03training.full_data.config_utils import grid_base, around
from step03training.full_data.analysis import LivePlotCallback

from shared.paths import vectors_file, index_file, scores_file, training_output_dir


def r2_metric(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[str, float, bool]:
    """Custom R2 metric for LightGBM evaluation."""
    return "r2", float(r2_score(y_true, y_pred)), True


def prepare_optimization_setup(
    base_cfg: Dict[str, Any],
) -> Tuple[Dict[str, Any], str]:
    param_grid: Dict[str, Any] = {}
    for key in grid_base.keys():
        if key not in base_cfg:
            raise ValueError(
                f"Base config missing required key '{key}' for optimization."
            )
        param_grid[key] = around(key, base_cfg[key])
    os.makedirs(training_output_dir, exist_ok=True)
    temp_model_base = os.path.join(training_output_dir, "temp_model")
    return (param_grid, temp_model_base)


def generate_combos(
    param_grid: Dict[str, Any], max_combos: int
) -> List[Dict[str, Any]]:
    keys = list(param_grid.keys())
    value_lists = [list(param_grid[k]) for k in keys]
    all_combos = [dict(zip(keys, vals)) for vals in product(*value_lists)]
    random.shuffle(all_combos)
    return list(islice(iter(all_combos), max_combos))


def evaluate_hyperparameter_combo(
    current_cfg: Dict[str, Any],
    temp_model_base: str,
    X: Optional[Any],
    y: Optional[Any],
) -> Tuple[float, float]:
    _, metrics = train_model(
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


def train_model_lightgbm_local(
    X_train: Optional[Any],
    X_test: Optional[Any],
    y_train: Optional[Any],
    y_test: Optional[Any],
    config_dict: Dict[str, Any],
    kept_features: Optional[np.ndarray] = None,
) -> Tuple[Any, Dict[str, Any]]:
    # Use LightGBM
    params = {}
    # Respect global device config, but mapping "cuda" -> "gpu" for LightGBM standard compatibility
    device_name = config["training"]["device"]
    if device_name == "cuda":
        device_name = "gpu"
    params["device_type"] = device_name
    params["random_state"] = config["training"]["random_state"]
    # Save user intent for progress bar, but silence LightGBM backend warnings
    user_verbosity = config["training"]["verbosity"]
    params["verbosity"] = -1

    # Dynamic parameter loading from grid_base
    for key in grid_base.keys():
        params[key] = config_dict[key]

    early_stopping_rounds = None
    if "early_stopping_rounds" in params:
        early_stopping_rounds = int(params.pop("early_stopping_rounds"))

    # Manually handle Target Transformation to support eval_set
    # LightGBM needs transformed targets for validation to match training
    transformer = PowerTransformer(method="yeo-johnson")

    # Reshape for transformer (requires 2D)
    y_train_2d = y_train.reshape(-1, 1)
    y_test_2d = y_test.reshape(-1, 1)

    # Fit on train, transform both
    y_train_trans = transformer.fit_transform(y_train_2d).ravel()
    y_test_trans = transformer.transform(y_test_2d).ravel()

    # Create base model
    model = lgb.LGBMRegressor(**params)

    # Setup callbacks for logging
    callbacks: List[Any] = []
    # Always suppress default logger to ensure clean output
    callbacks.append(lgb.log_evaluation(period=-1))

    if early_stopping_rounds is not None:
        callbacks.append(
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)
        )

    if user_verbosity >= 0:
        # Use tqdm progress bar
        pbar = tqdm(total=params["n_estimators"], desc="Training LightGBM")

        def pbar_callback(env):
            pbar.update(1)

        callbacks.append(pbar_callback)
        # Add live plotting callback (plots every 5s after first 10s)
        graph_path = training_output_dir + "graph.png"
        callbacks.append(
            LivePlotCallback(
                update_interval=60.0,
                min_time_before_first_plot=120.0,
                save_path=graph_path,
            )
        )

    start = time.time()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=".*X does not have valid feature names.*",
        )
        # Fit on transformed data
        model.fit(
            X_train,
            y_train_trans,
            eval_set=[(X_train, y_train_trans), (X_test, y_test_trans)],
            eval_names=["train", "valid"],
            eval_metric=["l2", "rmse", "l1", r2_metric],
            callbacks=callbacks,
        )
        if user_verbosity >= 0:
            pbar.close()
    training_time = time.time() - start

    # Construct final TransformedTargetRegressor with fitted components
    # This allows us to use .predict() normally (which inverses transform handles)
    final_model = TransformedTargetRegressor(regressor=model, transformer=transformer)
    # Manually set fitted attributes to bypass .fit()
    final_model.regressor_ = model
    final_model.transformer_ = transformer
    # Fix for manually initialized TransformedTargetRegressor
    final_model._training_dim = 1

    # Evaluate using the wrapper (handles inverse transform automatically)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=".*X does not have valid feature names.*",
        )
        y_pred = final_model.predict(X_test)

    mse_val = float(mean_squared_error(y_test, y_pred))
    metrics = {
        "r2": float(r2_score(y_test, y_pred)),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "mse": mse_val,
        "rmse": float(mse_val**0.5),
        "model_type": type(model).__name__,
        "training_time": training_time,
    }

    # Extract LightGBM-specific diagnostics
    if hasattr(model, "best_iteration_"):
        bi = int(getattr(model, "best_iteration_"))
        # If early stopping didn't trigger, best_iteration might be 0 or -1 depending on version,
        # or it might be n_estimators. If it's effectively unset, use n_estimators.
        if bi > 0:
            metrics["n_iter"] = bi
        else:
            metrics["n_iter"] = params["n_estimators"]

    # model.evals_result_ is populated if validation set is used
    if hasattr(model, "evals_result_"):
        evals = model.evals_result_
        # Structure: {'train': {'l2': [...]}, 'valid': {'l2': [...]}}
        # Try to find 'valid' then 'train', and 'l2' or 'rmse'
        target_set = (
            "valid" if "valid" in evals else ("train" if "train" in evals else None)
        )

        if target_set:
            # Get the first metric available
            metrics_dict = evals[target_set]
            if metrics_dict:
                first_metric = list(metrics_dict.keys())[0]
                metrics["loss_curve"] = metrics_dict[first_metric]
                metrics["loss_curve_length"] = len(metrics["loss_curve"])
                if "n_iter" not in metrics:
                    metrics["n_iter"] = len(metrics["loss_curve"])

                # Flag for notebook to know we have data
                metrics["has_loss_curve"] = True
                metrics["has_n_iter"] = True

    # Save the model
    # We save the wrapper so loading it works transparently
    if kept_features is not None:
        # If we have feature info, saved differently?
        # model_io.save_model usually handles just the model object + dicts.
        pass

    return final_model, metrics


def train_model(
    config_dict: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    kept_features: Optional[np.ndarray] = None,
) -> Tuple[Any, Dict[str, Any]]:

    if config["training"]["device"] != "cuda":
        raise ValueError("Training must use CUDA device.")

    test_size = float(config["training"]["test_size"])
    random_state = (config["training"]["random_state"])
    split_res = cast(
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        tuple(train_test_split(X, y, test_size=test_size, random_state=random_state)),
    )
    X_train, X_test, y_train, y_test = split_res

    return train_model_lightgbm_local(
        X_train,
        X_test,
        y_train,
        y_test,
        config_dict,
        kept_features=kept_features,
    )


def optimize_hyperparameters(
    base_cfg: Dict[str, Any],
    max_combos: int,
    X: Optional[Any] = None,
    y: Optional[Any] = None,
    strategy: str = "generic",
) -> List[Tuple[Dict[str, Any], Dict[str, float]]]:

    param_grid, temp_model_base = prepare_optimization_setup(base_cfg)
    print(f"Optimizing hyperparameters over grid: {param_grid}")

    combos = generate_combos(param_grid, max_combos)
    current_cfg = base_cfg.copy()
    results = []

    for i in range(len(combos)):
        combo = combos[i]
        print(
            f"Evaluating hyperparameter combo {i+1}/{len(combos)}, with params: {combo}"
        )
        merged = {**current_cfg, **combo}
        # print("merged:", merged)
        score, t_time = evaluate_hyperparameter_combo(merged, temp_model_base, X=X, y=y)
        print(f"r2={score:.4f}, time={t_time:.4f}s for Evaluated params {combo}")

        # Update Top (Always Global)
        # Re-fetch from config to ensure we have the latest global state
        training_section = config["training"]
        top_cfg = training_section["top"]
        current_top_score = (
            top_cfg["best_score"] if "best_score" in top_cfg else -1000000.0
        )

        if score > current_top_score:
            print(
                f"--> Found new TOP model! (old: {current_top_score:.4f}, new: {score:.4f})"
            )
            new_top = {**merged, "best_score": score, "training_time": t_time}
            top_cfg.update(new_top)
            # No need to manually assign back or save; AutoSaveDict handles it.
            current_top_score = score  # Update local var for the check below

        # Update Fastest (Track Restricted)
        if strategy == "FASTEST":
            fast_cfg = training_section["fastest"]
            current_fast_score = (
                fast_cfg["best_score"] if "best_score" in fast_cfg else -1000000.0
            )
            current_fast_time = (
                fast_cfg["training_time"] if "training_time" in fast_cfg else 999999.0
            )

            # Logic for "Fastest":
            # 1.  better score (>100.1% of current fastest score)
            # OR
            # 2. Faster time AND Acceptable score (>= 90% of TOP score)

            # Guard against negative/zero fast score for percentage check
            cond_a = False
            if current_fast_score > 0:
                cond_a = score >= 1.001 * current_fast_score
            else:
                # If current fastest score is negative/weird, any positive score is improvement
                cond_a = score > current_fast_score

            cond_b = (
                (t_time < current_fast_time)
                and (score >= 0.99 * current_top_score)
                and (score > current_fast_score)
            )

            if cond_a or cond_b:
                print(
                    f"--> Found new FASTEST model! (Score: {score:.4f}, Time: {t_time:.4f}s)"
                )
                new_fast = {**merged, "best_score": score, "training_time": t_time}
                fast_cfg.update(new_fast)
                # No need to manually assign back or save; AutoSaveDict handles it.

        # Update Slowest (Track Restricted)
        if strategy == "SLOWEST":
            slow_cfg = training_section["slowest"]
            if slow_cfg:
                current_slow_score = (
                    slow_cfg["best_score"] if "best_score" in slow_cfg else -1000000.0
                )
                if score > current_slow_score:
                    print(
                        f"--> Found new SLOWEST model! (Score: {score:.4f}, Time: {t_time:.4f}s)"
                    )
                    new_slow = {**merged, "best_score": score, "training_time": t_time}
                    slow_cfg.update(new_slow)

                    res_metrics = {"r2": score, "training_time": t_time}
                    results.append((merged, res_metrics))
                    if t_time > 300.0:
                        print(
                            "Slowest improved but still too slow; skipping remaining combos in this batch."
                        )
                        break

        res_metrics = {"r2": score, "training_time": t_time}
        results.append((merged, res_metrics))

    return results
