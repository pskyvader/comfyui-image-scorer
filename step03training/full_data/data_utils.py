import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import PolynomialFeatures
import gc
from tqdm import tqdm
from typing import Tuple
import os

from shared.io import load_jsonl
from shared.config import config
from shared.paths import training_output_dir, filtered_data, interaction_data


def load_training_vectors(vectors_path: str) -> np.ndarray:
    return load_jsonl(vectors_path, kind="vector")


def load_training_scores(scores_path: str) -> np.ndarray:
    return load_jsonl(scores_path, kind="scores")


def load_training_data(
    vectors_path: str, scores_path: str
) -> Tuple[np.ndarray, np.ndarray]:
    X = load_training_vectors(vectors_path)
    y = load_training_scores(scores_path)
    if len(X) != len(y):
        raise RuntimeError(
            f"Mismatched vector and score counts: vectors={len(X)}, scores={len(y)}; input files may be corrupted."
        )
    return (X, y)

def add_interaction_features(
    X: np.ndarray,
    y: np.ndarray,
    target_k: int = 500,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates and selects top K interaction features (x*y) using batched processing
    to avoid OOM. Concatenates them to X.
    Returns (X_final, selected_interaction_indices)
    Handles its own caching via 'interaction_data_cache.npz'.
    """
    if os.path.exists(interaction_data):
        try:
            data = np.load(interaction_data)
            if "X" in data and "interaction_indices" in data:
                print(f"Loading interaction data from cache: {interaction_data}")
                return data["X"], data["interaction_indices"]
        except Exception:
            pass

    n_features_in = X.shape[1]
    n_samples = X.shape[0]

    # Basic sizing check before proceeding
    # n*(n-1)/2 interactions

    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)

    # We only care about columns [n_features_in : ] from the poly expansion
    # Calculate number of interactions
    n_interactions = (n_features_in * (n_features_in - 1)) // 2

    if n_interactions == 0:
        return X, np.array([])

    # Batch Size Calculation (Target 4GB)
    BATCH_MEMORY_TARGET = 4 * 1024**3
    # Approx full poly width for memory calc
    n_features_total = n_features_in + n_interactions
    bytes_per_row = n_features_total * 8
    batch_size = int(BATCH_MEMORY_TARGET / bytes_per_row)
    if batch_size < 1:
        batch_size = 1
    if batch_size > 5000:
        batch_size = 5000

    # Accumulators for correlation calculation
    sum_x = np.zeros(n_interactions)
    sum_x_sq = np.zeros(n_interactions)
    sum_xy = np.zeros(n_interactions)
    sum_y = 0
    sum_y_sq = 0
    N = 0

    print(
        f"Scanning {n_interactions} potential interactions in batches of {batch_size}..."
    )

    # Pass 1: Stats
    with tqdm(total=n_samples, desc="Computing Correlations", unit="samples") as pbar:
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            X_batch = X[i:end_idx]
            y_batch = y[i:end_idx]

            # Generate Poly
            X_poly_full = poly.fit_transform(X_batch)
            # Extract only interactions
            X_inter_batch = X_poly_full[:, n_features_in:]

            # Stats
            sum_x += np.sum(X_inter_batch, axis=0)
            sum_x_sq += np.sum(X_inter_batch**2, axis=0)
            sum_xy += np.dot(X_inter_batch.T, y_batch)

            sum_y += np.sum(y_batch)
            sum_y_sq += np.sum(y_batch**2)
            N += end_idx - i

            del X_poly_full, X_inter_batch
            gc.collect()
            pbar.update(end_idx - i)

    # Compute Correlations (Pearson)
    numerator = (N * sum_xy) - (sum_x * sum_y)
    denominator_x = (N * sum_x_sq) - (sum_x**2)
    denominator_y = (N * sum_y_sq) - (sum_y**2)

    # Avoid div by zero / invalid sqrt
    denominator_x[denominator_x <= 0] = 1e-10
    denominator = np.sqrt(denominator_x * denominator_y)

    correlation = numerator / denominator
    f_scores = (correlation**2) / (1 - correlation**2 + 1e-10) * (N - 2)
    f_scores = np.nan_to_num(f_scores, nan=0.0)

    # Select Top K
    k = min(target_k, n_interactions)
    top_k_indices_local = np.argsort(f_scores)[-k:]
    top_k_indices_local = np.sort(top_k_indices_local)

    print(f"Selecting top {k} interaction features...")

    # Pass 2: Build
    X_interactions = np.zeros((n_samples, k), dtype=X.dtype)

    with tqdm(
        total=n_samples, desc="Building Interaction Matrix", unit="samples"
    ) as pbar:
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            X_batch = X[i:end_idx]

            X_poly_full = poly.fit_transform(X_batch)
            X_interactions[i:end_idx] = X_poly_full[:, n_features_in:][
                :, top_k_indices_local
            ]

            del X_poly_full
            gc.collect()
            pbar.update(end_idx - i)

    # Concatenate
    X_final = np.hstack([X, X_interactions])

    # --- Save Cache ---
    os.makedirs(training_output_dir, exist_ok=True)
    np.savez_compressed(
        interaction_data, X=X_final, interaction_indices=top_k_indices_local
    )
    print(f"Saved interaction data to cache: {interaction_data}")

    return X_final, top_k_indices_local


def filter_unused_features(
    X: np.ndarray,
    y: np.ndarray,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Trains a fast LightGBM model to identify and remove features with zero importance.
    Returns the filtered X dataset and the indices of the kept features.
    Handles its own caching via 'filtered_data_cache.npz'.
    """
    if verbose:
        print(f"Filtering features... Initial shape: {X.shape}")

    if os.path.exists(filtered_data):
        try:
            data = np.load(filtered_data)
            if "X" in data and "kept_indices" in data:
                print(f"Loading filtered data from cache: {filtered_data}")
                # Validate shape compatibility if possible or just trust cache
                return data["X"], data["kept_indices"]
        except Exception:
            pass

    device_name = config["training"]["device"]
    user_verbosity = config["training"]["verbosity"]
    if device_name == "cuda":
        device_name = "gpu"

    params = {
        "objective": "regression",
        "metric": "l2",
        "verbosity": -1,
        "n_estimators": 500,  # Enough to find gradients
        "learning_rate": 0.1,
        "min_child_samples": 1,
        "device_type": device_name,
    }
    model = lgb.LGBMRegressor(**params)

    # Setup callbacks for logging
    callbacks = []
    # Always suppress default logger to ensure clean output
    callbacks.append(lgb.log_evaluation(period=-1))

    if user_verbosity >= 0:
        # Use tqdm progress bar
        pbar = tqdm(total=params["n_estimators"], desc="Training LightGBM")

        def pbar_callback(env):
            pbar.update(1)

        callbacks.append(pbar_callback)

    model.fit(X, y, callbacks=callbacks)

    importances = model.feature_importances_
    mask = importances > 0
    kept_indices = np.where(mask)[0]

    if len(kept_indices) == 0:
        if verbose:
            print("Warning: No important features found. Keeping all features.")
        return X, np.arange(X.shape[1])

    X_filtered = X[:, mask]

    if verbose:
        n_dropped = X.shape[1] - X_filtered.shape[1]
        print(f"Dropped {n_dropped} unused features. New shape: {X_filtered.shape}")

    os.makedirs(training_output_dir, exist_ok=True)
    np.savez_compressed(filtered_data, X=X_filtered, kept_indices=kept_indices)
    if verbose:
        print(f"Saved filtered data to cache: {filtered_data}")

    return X_filtered, kept_indices

__all__ = [
    "load_training_data",
    "plot_scatter_comparison",
    "prepare_plot_data",
    "print_comparison_metrics",
    "filter_unused_features",
]
