import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import gc
from tqdm import tqdm
from typing import Tuple, Optional, Any, List
from pathlib import Path
import os

from shared.io import load_jsonl
from shared.config import config


def load_training_data(
    vectors_path: str, scores_path: str
) -> Tuple[np.ndarray, np.ndarray]:
    X = load_jsonl(vectors_path, kind="vector")
    y = load_jsonl(scores_path, kind="score")
    if len(X) != len(y):
        raise RuntimeError(
            f"Mismatched vector and score counts: vectors={len(X)}, scores={len(y)}; input files may be corrupted."
        )
    return (X, y)


def get_filtered_data(
    vectors_path: str, scores_path: str, cache_path: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads training data, filters unused features, adds interaction features, and caches the result.
    Invalidates cache if source file (vectors.jsonl) is newer than cache.
    Returns:
        X_final: Features (Filtered + Interactions)
        y: Target scores
        kept_indices: Indices of feature columns kept from original X
        interaction_indices: Indices of interaction features used relative to poly expansion of valid features
    """
    if cache_path is None:
         # Default to training/output/processed_data_cache.npz (renamed to reflect processing)
        root = Path(config["root"])
        cache_path = str(root / "training" / "output" / "processed_data_cache.npz")
    
    # Check cache validity
    cache_valid = False
    if os.path.exists(cache_path):
        try:
             # Check timestamp against vectors_path
            src_mtime = os.path.getmtime(vectors_path)
            cache_mtime = os.path.getmtime(cache_path)
            if src_mtime > cache_mtime:
                print("Source data (vectors.jsonl) is newer than cache. Invalidating cache.")
            else:
                cache_valid = True
        except Exception:
            pass # Fallback to rebuild

    if cache_valid:
        print(f"Loading processed data from cache: {cache_path}")
        try:
            data = np.load(cache_path)
            # Ensure integrity
            if "X" in data and "y" in data and "kept_indices" in data:
                print(f"Data ready (cached). Shape: {data['X'].shape}")
                # Load interactions if available, else empty (backward compatibility)
                inter_indices = data["interaction_indices"] if "interaction_indices" in data else np.array([])
                return data["X"], data["y"], data["kept_indices"], inter_indices
            else:
                print("Cache corrupted (missing keys). Rebuilding...")
        except Exception as e:
            print(f"Failed to load cache: {e}. Rebuilding...")

    print("Cache not found or invalid. Building processed dataset...")
    X, y = load_training_data(vectors_path, scores_path)
    
    # Step 1: Filter Unused Features
    X_filtered, kept_indices = filter_unused_features(X, y, vectors_path_for_hashing=vectors_path)
    
    # Step 2: Add Interaction Features (Top 500)
    print("Generating top interaction features...")
    X_final, interaction_indices = add_interaction_features(X_filtered, y, target_k=500, cache_source_path=vectors_path)
    
    # Ensure cache directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    
    np.savez_compressed(cache_path, X=X_final, y=y, kept_indices=kept_indices, interaction_indices=interaction_indices)
    print(f"Saved processed data to cache: {cache_path}")
    
    return X_final, y, kept_indices, interaction_indices


def add_interaction_features(X: np.ndarray, y: np.ndarray, target_k: int = 500, cache_source_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates and selects top K interaction features (x*y) using batched processing
    to avoid OOM. Concatenates them to X.
    Returns (X_final, selected_interaction_indices)
    Handles its own caching via 'interaction_data_cache.npz'.
    """
    # --- Cache Logic Start ---
    root = Path(config["root"])
    cache_path = str(root / "training" / "output" / "interaction_data_cache.npz")
    
    cache_valid = False
    if os.path.exists(cache_path):
        # Optional: Check validity against a source file (like filtered data cache)
        validity_check = True
        if cache_source_path and os.path.exists(cache_source_path):
             try:
                src_mtime = os.path.getmtime(cache_source_path)
                cache_mtime = os.path.getmtime(cache_path)
                if src_mtime > cache_mtime:
                    print("Source data newer than interaction cache. Rebuilding.")
                    validity_check = False
             except Exception:
                 pass
        
        if validity_check:
             try:
                data = np.load(cache_path)
                if "X" in data and "interaction_indices" in data:
                    print(f"Loading interaction data from cache: {cache_path}")
                    return data["X"], data["interaction_indices"]
             except Exception:
                pass
    # --- Cache Logic End ---

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
    if batch_size < 1: batch_size = 1
    if batch_size > 5000: batch_size = 5000 
    
    # Accumulators for correlation calculation
    sum_x = np.zeros(n_interactions)
    sum_x_sq = np.zeros(n_interactions)
    sum_xy = np.zeros(n_interactions)
    sum_y = 0
    sum_y_sq = 0
    N = 0
    
    print(f"Scanning {n_interactions} potential interactions in batches of {batch_size}...")
    
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
            sum_x_sq += np.sum(X_inter_batch ** 2, axis=0)
            sum_xy += np.dot(X_inter_batch.T, y_batch)
            
            sum_y += np.sum(y_batch)
            sum_y_sq += np.sum(y_batch ** 2)
            N += (end_idx - i)
            
            del X_poly_full, X_inter_batch
            gc.collect()
            pbar.update(end_idx - i)
            
    # Compute Correlations (Pearson)
    numerator = (N * sum_xy) - (sum_x * sum_y)
    denominator_x = (N * sum_x_sq) - (sum_x ** 2)
    denominator_y = (N * sum_y_sq) - (sum_y ** 2)
    
    # Avoid div by zero / invalid sqrt
    denominator_x[denominator_x <= 0] = 1e-10
    denominator = np.sqrt(denominator_x * denominator_y)
    
    correlation = numerator / denominator
    f_scores = (correlation ** 2) / (1 - correlation ** 2 + 1e-10) * (N - 2)
    f_scores = np.nan_to_num(f_scores, nan=0.0)
    
    # Select Top K
    k = min(target_k, n_interactions)
    top_k_indices_local = np.argsort(f_scores)[-k:]
    top_k_indices_local = np.sort(top_k_indices_local)
    
    print(f"Selecting top {k} interaction features...")
    
    # Pass 2: Build
    X_interactions = np.zeros((n_samples, k), dtype=X.dtype)
    
    with tqdm(total=n_samples, desc="Building Interaction Matrix", unit="samples") as pbar:
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            X_batch = X[i:end_idx]
            
            X_poly_full = poly.fit_transform(X_batch)
            X_interactions[i:end_idx] = X_poly_full[:, n_features_in:][:, top_k_indices_local]
            
            del X_poly_full
            gc.collect()
            pbar.update(end_idx - i)
            
    # Concatenate
    X_final = np.hstack([X, X_interactions])
    
    # --- Save Cache ---
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path, X=X_final, interaction_indices=top_k_indices_local)
    print(f"Saved interaction data to cache: {cache_path}")

    return X_final, top_k_indices_local


def filter_unused_features(
    X: np.ndarray, y: np.ndarray, vectors_path_for_hashing: Optional[str] = None, verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Trains a fast LightGBM model to identify and remove features with zero importance.
    Returns the filtered X dataset and the indices of the kept features.
    Handles its own caching via 'filtered_data_cache.npz'.
    """
    if verbose:
        print(f"Filtering features... Initial shape: {X.shape}")

    # --- Cache Logic Start ---
    root = Path(config["root"])
    cache_path = str(root / "training" / "output" / "filtered_data_cache.npz")
    
    cache_valid = False
    if os.path.exists(cache_path):
        # We need a source file to compare mtime against. 
        # If 'vectors_path_for_hashing' is passed (path to source vectors), we use that.
        # Otherwise, we can't reliably validate cache unless we trust existence implies validity.
        validity_check = True
        if vectors_path_for_hashing and os.path.exists(vectors_path_for_hashing):
             try:
                src_mtime = os.path.getmtime(vectors_path_for_hashing)
                cache_mtime = os.path.getmtime(cache_path)
                if src_mtime > cache_mtime:
                    if verbose: print("Source vectors newer than filter cache. Rebuilding.")
                    validity_check = False
             except Exception:
                 pass
        
        if validity_check:
            try:
                data = np.load(cache_path)
                if "X" in data and "kept_indices" in data:
                    print(f"Loading filtered data from cache: {cache_path}")
                    # Validate shape compatibility if possible or just trust cache
                    return data["X"], data["kept_indices"]
            except Exception:
                pass
    # --- Cache Logic End ---

    # Fast training configuration
    # Respect global device config, but mapping "cuda" -> "gpu" for LightGBM standard compatibility
    device_name = config["training"]["device"]
    user_verbosity = config["training"]["verbosity"]
    if device_name == "cuda":
        device_name = "gpu"

    params = {
        "objective": "regression",
        "metric": "l2",
        "verbosity": -1,
        "n_estimators": 100,  # Enough to find gradients
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

    model.fit(
        X,
        y,
        callbacks=callbacks
    )

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

    # --- Save Cache ---
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path, X=X_filtered, kept_indices=kept_indices)
    if verbose: print(f"Saved filtered data to cache: {cache_path}")

    return X_filtered, kept_indices


def plot_scatter_comparison(
    y_plot: np.ndarray, p_plot: np.ndarray, plot: bool = True
) -> None:
    _, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(
        y_plot, p_plot, alpha=0.7, s=30, edgecolors="k", linewidths=0.2, zorder=5
    )
    ymin = float(min(np.min(y_plot), np.min(p_plot)))
    ymax = float(max(np.max(y_plot), np.max(p_plot)))
    if not np.isfinite(ymin) or not np.isfinite(ymax):
        print("Could not determine plot range; skipping perfect-prediction line.")
    else:
        margin = max((ymax - ymin) * 0.02, 1e-06)
        ax.plot(
            [ymin - margin, ymax + margin],
            [ymin - margin, ymax + margin],
            "r--",
            label="perfect prediction",
            linewidth=2.0,
            zorder=10,
        )
        ax.set_xlim(ymin - margin, ymax + margin)
        ax.set_ylim(ymin - margin, ymax + margin)
        ax.set_aspect("equal", adjustable="box")
    ax.legend()
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted (sample)")
    ax.grid(True)
    if plot:
        plt.show()


def prepare_plot_data(y: Any, preds: Any) -> Tuple[np.ndarray, np.ndarray]:
    y_sample = np.asarray(y[:100]).ravel()
    preds = np.asarray(preds).ravel()
    mask = np.isfinite(y_sample) & np.isfinite(preds)
    if not mask.any():
        print("No finite prediction/data pairs to plot; skipping compare plot.")
        return (np.array([]), np.array([]))
    return (y_sample[mask], preds[mask])


def print_comparison_metrics(y: Any, preds: Any, metrics: Any) -> None:
    y_sample = np.asarray(y).ravel()
    preds = np.asarray(preds).ravel()
    mask_all = np.isfinite(y_sample) & np.isfinite(preds)
    if mask_all.any():
        y_eval = y_sample[mask_all]
        p_eval = preds[mask_all]
        sample_r2 = float(r2_score(y_eval, p_eval))
        print(f"Comparison metrics (sample): r2={sample_r2:.4f}, n={len(y_eval)}")
    if metrics is not None and "r2" in metrics:
        print(f"Stored metrics: r2={float(metrics['r2']):.4f}")


__all__ = [
    "load_training_data",
    "plot_scatter_comparison",
    "prepare_plot_data",
    "print_comparison_metrics",
    "filter_unused_features",
]
