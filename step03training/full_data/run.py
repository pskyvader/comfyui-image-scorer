from typing import Any, Dict, Tuple, cast, List, Optional, Set
import os
import time
import warnings
import random
from itertools import product, islice

import joblib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import PowerTransformer

from shared.config import config

# save_config is removed, using direct assignment
from training.helpers import resolve_path
from training.data_utils import (
    load_training_data,
    prepare_plot_data,
    print_comparison_metrics,
    plot_scatter_comparison,
    filter_unused_features,
    get_filtered_data,
)
from training.model_io import load_model_diagnostics
import lightgbm as lgb
# Note: onnxruntime is imported lazily where needed to avoid import issues on newer Python versions

from training.config_utils import grid_base, around


def r2_metric(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[str, float, bool]:
    """Custom R2 metric for LightGBM evaluation."""
    return "r2", float(r2_score(y_true, y_pred)), True


class LivePlotCallback:
    """Callback to plot training progress in real-time inside Jupyter notebooks."""

    def __init__(
        self,
        update_interval: float = 5.0,
        min_time_before_first_plot: float = 10.0,
        save_path: Optional[str] = None,
    ):
        self.update_interval = update_interval
        self.min_time = min_time_before_first_plot
        self.save_path = save_path
        if self.save_path:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        self.start_time = time.time()
        self.last_plot_time = 0.0
        self.history: Dict[str, List[float]] = {}  # Generalized history storage

    def __call__(self, env: Any) -> None:
        now = time.time()
        elapsed = now - self.start_time

        # Collect metrics
        # env.evaluation_result_list looks like [('train', 'l2', 0.5, False), ('valid', 'l2', 0.6, False), ...]
        for data_name, eval_name, result, _ in env.evaluation_result_list:
            key = f"{data_name}_{eval_name}"
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(result)

        if elapsed < self.min_time:
            return

        if now - self.last_plot_time < self.update_interval:
            return

        self.last_plot_time = now

        try:
            from IPython.display import clear_output

            clear_output(wait=True)

            # Group metrics by type (l2, rmse, l1)
            metrics_found: Set[str] = set()
            for key in self.history.keys():
                parts = key.split("_")
                if len(parts) >= 2:
                    metrics_found.add(parts[1])

            num_metrics = len(metrics_found)
            if num_metrics == 0:
                return

            fig, axes = plt.subplots(1, num_metrics, figsize=(5 * num_metrics, 5))
            if num_metrics == 1:
                axes = [axes]

            for i, metric in enumerate(sorted(list(metrics_found))):
                ax = axes[i]
                if f"train_{metric}" in self.history:
                    data = self.history[f"train_{metric}"]
                    ax.plot(data, label=f"Train {metric}")
                    if data:
                        ax.annotate(
                            f"{data[-1]:.4f}",
                            xy=(len(data) - 1, data[-1]),
                            xytext=(5, 0),
                            textcoords="offset points",
                        )
                if f"valid_{metric}" in self.history:
                    data = self.history[f"valid_{metric}"]
                    ax.plot(data, label=f"Valid {metric}")
                    if data:
                        ax.annotate(
                            f"{data[-1]:.4f}",
                            xy=(len(data) - 1, data[-1]),
                            xytext=(5, 0),
                            textcoords="offset points",
                        )
                ax.set_xlabel("Iteration")
                ax.set_title(f"{metric.upper()} Progress")
                ax.legend()
                ax.grid(True)

            plt.suptitle(f"Training Progress (Elapsed: {elapsed:.1f}s)")
            plt.tight_layout()

            if self.save_path:
                try:
                    plt.savefig(self.save_path)
                except Exception:
                    pass

            plt.show()
        except ImportError:
            pass
        except Exception:
            pass


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
    out_dir_val = config["training"]["output_dir"]
    out_dir = resolve_path(out_dir_val)
    os.makedirs(out_dir, exist_ok=True)
    temp_model_base = os.path.join(out_dir, "temp_model")
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
        temp_model_base,
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
    model_path: str,
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
    callbacks: List[str, Any] = []
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
        graph_path = resolve_path(config["training"]["live_graph_path"])
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

    # Save transformer parameters for ONNX post-processing
    if hasattr(transformer, "lambdas_"):
        metrics["target_transform_lambdas"] = transformer.lambdas_
        if hasattr(transformer, "_scaler"):
            metrics["target_transform_mean"] = transformer._scaler.mean_
            metrics["target_transform_scale"] = transformer._scaler.scale_

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

    # We return the WRAPPED model
    return final_model, metrics

    # We return the WRAPPED model
    return final_model, metrics

    # Extract LightGBM-specific diagnostics
    if hasattr(model, "best_iteration_"):
        bi = int(getattr(model, "best_iteration_"))
        if bi > 0:
            metrics["n_iter"] = bi
        elif "loss_curve_length" in metrics:
            metrics["n_iter"] = metrics["loss_curve_length"]
        else:
            metrics["n_iter"] = params["n_estimators"]

    # model.evals_result_ may be present
    if hasattr(model, "evals_result_"):
        evals_result = getattr(model, "evals_result_")
        # try to extract train or validation loss sequences
        # evals_result_ is dict(namespace -> metric -> list)

        # LightGBM usually stores 'l2' or 'rmse' etc.
        # evals_result = {'train': {'l2': [...]}, 'valid': {'l2': [...]}}
        # We prefer validation loss if available
        loss_curve = None
        for ns in ["valid", "test", "train"]:
            if ns in evals_result:
                for metric_name in evals_result[ns]:
                    loss_curve = evals_result[ns][metric_name]
                    break
            if loss_curve:
                break

        if loss_curve:
            metrics["loss_curve"] = loss_curve
            metrics["loss_curve_length"] = len(loss_curve)
            metrics["has_loss_curve"] = True

    if kept_features is not None:
        metrics["kept_features"] = kept_features

    return model, metrics


def train_model(
    model_path: str,
    config_dict: Dict[str, Any],
    X: Optional[Any] = None,
    y: Optional[Any] = None,
    kept_features: Optional[np.ndarray] = None,
) -> Tuple[Any, Dict[str, Any]]:

    if config["training"]["device"] != "cuda":
        raise ValueError("Training must use CUDA device.")

    if not os.path.isabs(model_path) and os.path.dirname(model_path) in ("", "."):
        model_path = os.path.join(out_dir, model_path)
    model_path = os.path.abspath(os.path.normpath(model_path))

    if X is None or y is None:
        vectors_path = resolve_path(config["vectors_file"])
        scores_path = resolve_path(config["scores_file"])
        X, y = load_training_data(vectors_path, scores_path)

    test_size = float(config["training"]["test_size"])
    random_state = int(config["training"]["random_state"])
    split_res = cast(
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        tuple(train_test_split(X, y, test_size=test_size, random_state=random_state)),
    )
    X_train, X_test, y_train, y_test = split_res

    return train_model_lightgbm_local(
        model_path,
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


def compare_model_vs_data(
    model_path: str,
    vectors_path: str,
    scores_path: str,
    plot: bool = True,
    model: Any | None = None,
) -> None:
    """Compare a trained model vs data.

    This function requires a trained model to be supplied either as the `model`
    argument (in-memory estimator) or as a saved model file on disk. If no
    trained model is available, a RuntimeError is raised. This avoids silently
    re-training the model during comparison.
    """
    # Load data using the full pipeline (including filtering and interactions) to ensure consistency with trained model
    X, y, _, _ = get_filtered_data(vectors_path, scores_path)

    x_sample = X[:100]
    # align y to the sample used for predictions
    y_sample = y[: len(x_sample)]

    # If no model provided, attempt to load from disk (common extensions).
    if model is None:
        # User requested ONNX priority despite complications with target transforms.
        # We will attempt to handle target transform manually if ONNX is loaded.
        candidates = [
            model_path + ".onnx",
            model_path + ".joblib",
            model_path + ".pkl",
            model_path,
        ]
        found = None
        for p in candidates:
            try:
                if os.path.exists(p):
                    found = p
                    break
            except Exception:
                continue

        if found is None:
            raise RuntimeError(
                "No trained model provided and no saved model file found. "
                "Please pass a trained model object via the `model` argument or save the trained model to disk (e.g., model_path.onnx)."
            )

        print(f"Loading trained model from: {found}")

        # If ONNX, use onnxruntime for inference
        if found.lower().endswith(".onnx"):
            import onnxruntime as ort

            sess = ort.InferenceSession(
                found, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )

            # Check for diagnostics to apply inverse transform
            transformer_params = {}
            try:
                data = load_model_diagnostics(model_path)
                if data:
                    # Diagnostics might be nested in 'metrics' or top level depending on how saved
                    # current impl in save_model saves dict as **additional_data.
                    # so keys should be top level in .npz
                    if "target_transform_lambdas" in data:
                        transformer_params["lambdas"] = data["target_transform_lambdas"]
                        transformer_params["mean"] = data.get("target_transform_mean")
                        transformer_params["scale"] = data.get("target_transform_scale")
                    elif "metrics" in data:
                        # Handle nested metrics if that happens
                        m = data["metrics"]
                        if isinstance(m, dict) and "target_transform_lambdas" in m:
                            transformer_params["lambdas"] = m[
                                "target_transform_lambdas"
                            ]
                            transformer_params["mean"] = m.get("target_transform_mean")
                            transformer_params["scale"] = m.get(
                                "target_transform_scale"
                            )
            except Exception as e:
                print(
                    f"Warning: Could not load diagnostics for ONNX transform check: {e}"
                )

            # Prepare input name and run
            input_name = sess.get_inputs()[0].name

            def onnx_predict(X_arr: np.ndarray) -> np.ndarray:
                # Ensure float32
                Xf = X_arr.astype(np.float32)
                out = sess.run(None, {input_name: Xf})
                y_pred = np.asarray(out[0]).ravel()

                # Apply Inverse Transform if parameters found
                if "lambdas" in transformer_params:
                    # Yeo-Johnson Inverse Application

                    # 1. Inverse Standardize
                    if (
                        transformer_params.get("mean") is not None
                        and transformer_params.get("scale") is not None
                    ):
                        mean = transformer_params["mean"]
                        scale = transformer_params["scale"]
                        y_pred = y_pred * scale + mean

                    # 2. Inverse Yeo-Johnson
                    lmbda = transformer_params["lambdas"]
                    if isinstance(lmbda, np.ndarray) and lmbda.size == 1:
                        lmbda = lmbda.item()

                    y_inv = np.zeros_like(y_pred)

                    if abs(lmbda) > 1e-7:
                        pos_mask = y_pred >= 0
                        neg_mask = ~pos_mask

                        # Positive
                        y_pos = y_pred[pos_mask]
                        base_pos = y_pos * lmbda + 1
                        base_pos = np.maximum(base_pos, 1e-9)
                        y_inv[pos_mask] = np.power(base_pos, 1.0 / lmbda) - 1

                        # Negative
                        y_neg = y_pred[neg_mask]
                        l2 = 2 - lmbda
                        base_neg = 1 - y_neg * l2
                        base_neg = np.maximum(base_neg, 1e-9)
                        y_inv[neg_mask] = 1 - np.power(base_neg, 1.0 / l2)
                    else:
                        pos_mask = y_pred >= 0
                        neg_mask = ~pos_mask
                        y_inv[pos_mask] = np.exp(y_pred[pos_mask]) - 1
                        y_inv[neg_mask] = 1 - np.exp(-y_pred[neg_mask])

                    return y_inv

                return y_pred

            model = type("ONNXWrapper", (), {"predict": staticmethod(onnx_predict)})()
        else:
            try:
                model = joblib.load(found)
            except Exception as e:
                raise RuntimeError(f"Failed to load trained model from '{found}': {e}")
    else:
        print("Using provided in-memory model for comparison.")

    # Use the supplied/loaded model for predictions (do not retrain here).
    preds = model.predict(x_sample)
    # metrics may be available from saved diagnostics; attempt to load them
    try:
        metrics = None
        data = load_model_diagnostics(model_path)
        if data is not None and "metrics" in data:
            metrics = data["metrics"]
    except Exception:
        metrics = None

    print_comparison_metrics(y_sample, preds, metrics)
    y_plot, p_plot = prepare_plot_data(y_sample, preds)
    if y_plot.size > 0 and p_plot.size > 0:
        plot_scatter_comparison(y_plot, p_plot, plot)


def plot_loss_curve(model_path: str, metrics: Dict[str, Any] | None = None) -> None:
    try:
        loss = None
        if (
            metrics is not None
            and "loss_curve" in metrics
            and (metrics["loss_curve"] is not None)
        ):
            loss = np.asarray(metrics["loss_curve"], dtype=float)
        data = load_model_diagnostics(model_path)
        if data is not None and "loss_curve" in data:
            loss = np.asarray(data["loss_curve"], dtype=float)
        if loss is None:
            print("No loss curve available to plot.")
            return
        loss = loss.ravel()
        if loss.size == 0:
            print("Loss curve empty; nothing to plot.")
            return
        plt.figure(figsize=(6, 4))
        plt.plot(np.arange(1, loss.size + 1), loss, "-o", linewidth=1)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.grid(True)
        plt.show()
    except Exception as e:
        print("Failed to plot loss curve:", e)


def run_training(
    *, model_path: str, vectors_path: str, scores_path: str, config_dict: Dict[str, Any]
) -> Tuple[Any, Dict[str, float]]:
    print(
        f"Starting training with:\n  model_path: {model_path}\n  vectors_path: {vectors_path}\n  scores_path: {scores_path}"
    )

    # Load and filter data (optimization step)
    X, y = load_training_data(vectors_path, scores_path)
    X, kept_features = filter_unused_features(X, y, feature_names=None, verbose=True)

    model, metrics = train_model(
        model_path, config_dict, X=X, y=y, kept_features=kept_features
    )

    print("Training complete. Evaluation metrics:")
    for k, v in metrics.items():
        try:
            print(f"  {k}: {v:.4f}")
        except Exception:
            print("  {}: {}".format(k, v))

    if "n_iter" in metrics:
        print(f"Training iterations: {metrics['n_iter']}")
    if "loss_curve_length" in metrics:
        print(f"Loss curve length: {metrics['loss_curve_length']}")

    return (model, metrics)
