import numpy as np
from pathlib import Path
from ..paths import models_dir, training_model, filtered_data
from ..helpers import load_model, load_model_diagnostics, get_param
# from sklearn.compose._target import TransformedTargetRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import PowerTransformer
import warnings


class ScorerModel:
    def __init__(self):
        self.model_dir = Path(models_dir)
        self.diagnostics_path = training_model

        print(f"Loading scoring model from {self.model_dir}...")
        self.model: TransformedTargetRegressor = load_model(training_model)
        try:
            cache_data = np.load(filtered_data)
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

        diag_data = load_model_diagnostics(training_model)

        self.transformer_params["lambdas"] = get_param(
            "target_transform_lambdas", diag_data
        )
        print(
            f"self.transformer_params[lambdas] type: {type(self.transformer_params["lambdas"])}"
        )
        self.transformer_params["mean"] = get_param("target_transform_mean", diag_data)
        print(
            f"self.transformer_params[mean] type: {type(self.transformer_params["mean"])}"
        )
        self.transformer_params["scale"] = get_param(
            "target_transform_scale", diag_data
        )
        print(
            f"self.transformer_params[scale] type: {type(self.transformer_params["scale"])}"
        )

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
        Xf = X_raw.astype(np.float32)
        print(f"Xf shape: {Xf.shape}")
        preds = self.model.predict(Xf)
        print(f"preds: {preds}")

        # # Construct final TransformedTargetRegressor with fitted components
        # # This allows us to use .predict() normally (which inverses transform handles)
        # transformer = PowerTransformer(method="yeo-johnson")
        # final_model = TransformedTargetRegressor(
        #     regressor=self.model, transformer=transformer
        # )
        # # Manually set fitted attributes to bypass .fit()
        # final_model.regressor_ = self.model
        # final_model.transformer_ = transformer
        # # Fix for manually initialized TransformedTargetRegressor
        # final_model._training_dim = 1

        # # Evaluate using the wrapper (handles inverse transform automatically)
        # with warnings.catch_warnings():
        #     warnings.filterwarnings(
        #         "ignore",
        #         category=UserWarning,
        #         message=".*X does not have valid feature names.*",
        #     )
        #     preds = final_model.predict(Xf)

        # print(f"final_model preds shape: {preds}")
        
        y_sample = np.asarray(preds).ravel()
        print(f"y_sample shape: {y_sample.shape}")

        print(f"predicted transformed scores: {y_sample}")
        final_scores = self._inverse_transform_score(y_sample)
        print(f"final scores: {final_scores}")
        return final_scores.tolist()
