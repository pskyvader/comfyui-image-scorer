from typing import List
import numpy as np
try:
    import onnxruntime as ort
except Exception:
    ort = None
from pathlib import Path

import sklearn.preprocessing


class ScorerModel:
    """
    Wraps the ONNX model and the interaction generation logic.
    """

    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self.onnx_path = self.model_dir / "model.onnx"
        self.diagnostics_path = self.model_dir / "model.npz"
        self.filter_cache_path = self.model_dir / "processed_data_cache.npz"

        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found at {self.onnx_path}")

        # print(f"Loading scoring model from {self.model_dir}...")

        # Load ONNX Session
        try:
            self.sess = ort.InferenceSession(
                str(self.onnx_path),
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
        except Exception as e:
            self.sess = ort.InferenceSession(
                str(self.onnx_path), providers=["CPUExecutionProvider"]
            )

        self.input_name = self.sess.get_inputs()[0].name

        # Load Metadata (Filtering Mask & Interaction Indices)
        if not self.filter_cache_path.exists():
            raise FileNotFoundError(
                "Processing cache (processed_data_cache.npz) needed for feature definitions not found."
            )

        try:
            cache_data = np.load(self.filter_cache_path)
            self.kept_indices = cache_data["kept_indices"]
            self.interaction_indices = (
                cache_data["interaction_indices"]
                if "interaction_indices" in cache_data
                else np.array([])
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load feature processing metadata: {e}")

        # Load Target Transform Parameters (Yeo-Johnson)
        self.transformer_params = {}
        if self.diagnostics_path.exists():
            try:
                diag_data = np.load(self.diagnostics_path, allow_pickle=True)

                def get_param(key, data):
                    if key in data:
                        return data[key]
                    if "metrics" in data:
                        m = data["metrics"]
                        if hasattr(m, "item"):
                            m = m.item()
                        if isinstance(m, dict) and key in m:
                            return m[key]
                    return None

                self.transformer_params["lambdas"] = get_param(
                    "target_transform_lambdas", diag_data
                )
                self.transformer_params["mean"] = get_param(
                    "target_transform_mean", diag_data
                )
                self.transformer_params["scale"] = get_param(
                    "target_transform_scale", diag_data
                )
            except Exception:
                pass

    def _transform_input(self, X_raw: np.ndarray) -> np.ndarray:
        # Ensure 2D
        if X_raw.ndim == 1:
            X_raw = X_raw.reshape(1, -1)

        # 1. Feature Filtering
        print(f"Raw input shape: {X_raw.shape}")
        print(f"Kept indices: {self.kept_indices}")
        if self.kept_indices is not None:
            X_filtered = X_raw[:, self.kept_indices]
        else:
            X_filtered = X_raw

        print(f"After filtering shape: {X_filtered.shape}")
        # 2. Add Interactions
        if len(self.interaction_indices) > 0:
            n_samples, n_features = X_filtered.shape

            # Simple approach: Generate all interactions, then slice.
            # 8MB memory overhead for 1 sample is fine.
            try:

                poly = sklearn.preprocessing.PolynomialFeatures(
                    degree=2, include_bias=False, interaction_only=True
                )
                X_poly = poly.fit_transform(X_filtered)

                n_base = X_filtered.shape[1]
                base_interactions = X_poly[:, n_base:]
                selected_interactions = base_interactions[:, self.interaction_indices]

                X_final = np.hstack([X_filtered, selected_interactions])

            except ImportError:
                # Replicate logic manually if sklearn missing
                # n_features ~1300. interactions = n*(n-1)/2.
                # We only need specific indices.
                # Without mapping k -> (i,j), we must generate all iteratively or check mapping.
                # Assuming sklearn is available for now.
                # If not, User needs to install it or we implement the iterator.
                raise ImportError("scikit-learn is required for interaction features.")
        else:
            X_final = X_filtered

        return X_final.astype(np.float32)

    def _inverse_transform_score(self, y_pred_raw: np.ndarray) -> np.ndarray:
        """Applies inverse Yeo-Johnson + Inverse Standardization."""
        y_inv = y_pred_raw.copy()

        # 1. Inverse Standardize
        if (
            self.transformer_params.get("mean") is not None
            and self.transformer_params.get("scale") is not None
        ):
            mean = self.transformer_params["mean"]
            scale = self.transformer_params["scale"]
            y_inv = y_inv * scale + mean

        # 2. Inverse Yeo-Johnson
        lmbda = self.transformer_params.get("lambdas")
        if lmbda is not None:
            if isinstance(lmbda, np.ndarray) and lmbda.size == 1:
                lmbda = lmbda.item()

            if abs(lmbda) > 1e-7:
                pos_mask = y_inv >= 0
                neg_mask = ~pos_mask

                if np.any(pos_mask):
                    y_p = y_inv[pos_mask]
                    base = np.maximum(y_p * lmbda + 1, 1e-9)
                    y_inv[pos_mask] = np.power(base, 1.0 / lmbda) - 1

                if np.any(neg_mask):
                    y_n = y_inv[neg_mask]
                    l2 = 2 - lmbda
                    base = np.maximum(1 - y_n * l2, 1e-9)
                    y_inv[neg_mask] = 1 - np.power(base, 1.0 / l2)
            else:
                pos_mask = y_inv >= 0
                y_inv[pos_mask] = np.exp(y_inv[pos_mask]) - 1
                y_inv[~pos_mask] = 1 - np.exp(-y_inv[~pos_mask])

        return y_inv

    def predict(self, X_raw: np.ndarray) -> np.ndarray:
        """Returns single float score for first sample."""
        # X_processed = self._transform_input(X_raw)
        # X_processed: List[np.ndarray] = []
        # for x in X_raw:
        #     if x.ndim == 1:
        #         x = x.reshape(1, -1)
        #     X_processed.append(x.astype(np.float32))

        # X_raw: np.ndarray = np.stack(X_raw)

        Xf = X_raw.astype(np.float32)
        out = self.sess.run(None, {self.input_name: Xf})
        y_pred_trans = np.asarray(out[0]).ravel()

        print(f"predicted transformed scores: {y_pred_trans}")
        final_scores = self._inverse_transform_score(y_pred_trans)
        print(f"final scores: {final_scores}")
        return final_scores.tolist()
        # Inverse Transform
        final_scores: List[float] = []
        for pred in y_pred_trans:
            print(f"predicted transformed score: {pred}")
            final_scores.append(self._inverse_transform_score(pred))
        print(f"final scores: {final_scores}")
        return final_scores
