from typing import Any, Dict, Tuple, cast, List
import time
import warnings
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
    accuracy_score,
    log_loss,
)

from sklearn.model_selection import train_test_split

# from sklearn.compose import TransformedTargetRegressor
# from sklearn.preprocessing import PowerTransformer

from shared.config import config

import lightgbm as lgb
from external_modules.step03training.full_data.config_utils import grid_base
from external_modules.step03training.full_data.analysis import LivePlotCallback

from shared.paths import models_dir


class ModelTrainer:
    def __init__(self) -> None:
        self.training_model = None
        self.eval_metrics: List[Any] = []
        # self.final_model = None
        self.n_estimators = None
        self.early_stopping_rounds = None
        self.user_verbosity = config["training"]["verbosity"]
        self.callbacks = None
        # self.transformer = PowerTransformer(method="yeo-johnson")
        self.result_metrics = {}

    def r2_metric(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Tuple[str, float, bool]:
        """Custom R2 metric for LightGBM evaluation."""
        return "r2", float(r2_score(y_true, y_pred)), True

    def create_training_model(self, config_dict: Dict[str, Any]):

        # Base LightGBM parameters
        params: Dict[str, Any] = {
            "random_state": config["training"]["random_state"],
            "verbosity": -1,
            "objective": config["training"]["objective"],
            "device_type": "gpu",
        }
        # Dynamically load grid parameters
        for key in grid_base.keys():
            if key in config_dict:
                params[key] = config_dict[key]
        self.n_estimators = int(params["n_estimators"])
        if "early_stopping_rounds" in params:
            self.early_stopping_rounds = int(params.pop("early_stopping_rounds"))
        objective = params["objective"]

        # Ranking
        if objective == "lambdarank":
            params["label_gain"] = [0, 1, 3, 7, 15, 31]
            self.training_model = lgb.LGBMRanker(**params)
            self.eval_metrics = ["ndcg"]
        # Classification
        elif objective in {"multiclass", "binary"}:
            self.training_model = lgb.LGBMClassifier(**params)
            if objective == "multiclass":
                self.eval_metrics = ["multi_logloss", "multi_error"]
            else:
                self.eval_metrics = ["binary_logloss", "auc"]
        # Regression
        else:
            self.training_model = lgb.LGBMRegressor(**params)
            self.eval_metrics = ["l2", "rmse", "l1", self.r2_metric]

    def create_callbacks(self, pbar: Any) -> None:
        # Setup callbacks for logging
        callbacks: List[Any] = []
        # Always suppress default logger to ensure clean output
        callbacks.append(lgb.log_evaluation(period=-1))

        if self.early_stopping_rounds is not None:
            callbacks.append(
                lgb.early_stopping(
                    stopping_rounds=self.early_stopping_rounds, verbose=False
                )
            )

        if self.user_verbosity >= 0:
            # Use tqdm progress bar
            def pbar_callback(_):
                pbar.update(1)

            callbacks.append(pbar_callback)
            # Add live plotting callback (plots every 5s after first 10s)
            graph_path = models_dir + "graph.png"
            callbacks.append(
                LivePlotCallback(
                    update_interval=60.0,
                    min_time_before_first_plot=120.0,
                    save_path=graph_path,
                )
            )

        self.callbacks = callbacks

    # def create_final_model(self):
    #     """Construct final TransformedTargetRegressor with fitted components
    #     This allows us to use .predict() normally (which inverses transform handles)
    #     transformer = PowerTransformer(method="yeo-johnson")
    #     """
    #     self.final_model = TransformedTargetRegressor(
    #         regressor=self.training_model, transformer=self.transformer
    #     )
    #     # Manually set fitted attributes to bypass .fit()
    #     self.final_model.regressor_ = self.training_model
    #     self.final_model.transformer_ = self.transformer
    #     # Fix for manually initialized TransformedTargetRegressor
    #     self.final_model._training_dim = 1

    def create_metrics(
        self, y_test: np.ndarray, y_pred: np.ndarray, training_time: float
    ):
        model = self.training_model
        model_type = type(model).__name__

        self.result_metrics: Dict[str, Any] = {
            "model_type": model_type,
            "training_time": training_time,
        }

        # Iterations
        best_iter = (
            int(getattr(model, "best_iteration_"))
            if hasattr(model, "best_iteration_")
            else -1
        )
        self.result_metrics["n_iterations"] = (
            best_iter if best_iter > 0 else self.n_estimators
        )

        # Objective-aware external metrics
        objective = model.get_params().get("objective")

        if objective in {"regression", "regression_l2", "l2"}:
            mae = float(mean_absolute_error(y_test, y_pred))
            rmse = float(root_mean_squared_error(y_test, y_pred))

            self.result_metrics.update(
                {
                    "mae": mae,
                    "rmse": rmse,
                }
            )

        elif objective in {"multiclass", "binary"}:
            # Classification metrics
            if objective == "binary":
                predictions = (y_pred > 0.5).astype(int)
            else:
                predictions = np.argmax(y_pred, axis=1)

            acc = float(accuracy_score(y_test, predictions))
            self.result_metrics["accuracy"] = acc

            # log loss only if probabilities available
            try:
                ll = float(log_loss(y_test, y_pred))
                self.result_metrics["log_loss"] = ll
            except Exception:
                pass

        # Ranking usually uses internal metrics only

        # Extract LightGBM evaluation results
        evals = getattr(model, "evals_result_", None)
        metric_dict = None
        if evals:
            target_set = (
                "valid" if "valid" in evals else "train" if "train" in evals else None
            )
            if target_set:
                metric_dict = evals[target_set]

        primary_metric = self.eval_metrics[0]
        if metric_dict:
            # Store all metrics from LightGBM
            for metric_name, values in metric_dict.items():
                self.result_metrics[f"{metric_name}_curve"] = values
                self.result_metrics[f"{metric_name}_final"] = values[-1]

            # Define main score = first metric in eval_metric list
            if primary_metric in metric_dict:
                self.result_metrics["score"] = metric_dict[primary_metric][-1]

    # def apply_transform(self, y_train, y_test  ):
    #     # Manually handle Target Transformation to support eval_set
    #     # LightGBM needs transformed targets for validation to match training

    #     # Reshape for transformer (requires 2D)
    #     y_train_2d = y_train.reshape(-1, 1)
    #     y_test_2d = y_test.reshape(-1, 1)

    #     # Fit on train, transform both
    #     y_train_trans = self.transformer.fit_transform(y_train_2d).ravel()
    #     y_test_trans = self.transformer.transform(y_test_2d).ravel()
    #     return y_train_trans, y_test_trans

    def train_model(
        self,
        config_dict: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[Any, Dict[str, Any]]:

        if config["training"]["device"] != "cuda":
            raise ValueError("Training must use CUDA device.")

        test_size = float(config["training"]["test_size"])
        random_state = config["training"]["random_state"]
        split_res = cast(
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
            tuple(
                train_test_split(X, y, test_size=test_size, random_state=random_state)
            ),
        )
        x_train, x_test, y_train, y_test = split_res
        #  y_train_trans, y_test_trans=self.apply_transform(y_train, y_test )

        self.create_training_model(config_dict)
        pbar = tqdm(total=self.n_estimators, desc="Training LightGBM")
        self.create_callbacks(pbar)
        start = time.time()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message=".*X does not have valid feature names.*",
            )

            parameters: Dict[str, Any] = {
                "X": x_train,
                "y": y_train,
                "eval_set": [(x_train, y_train), (x_test, y_test)],
                "eval_names": ["train", "valid"],
                "eval_metric": self.eval_metrics,
                "callbacks": self.callbacks,
            }

            if isinstance(self.training_model, lgb.LGBMRanker):
                self.training_model.fit(
                    **parameters,
                    group=[len(y_train)],
                    eval_group=[[len(y_train)], [len(y_test)]],
                )
            else:
                self.training_model.fit(**parameters)

            if self.user_verbosity >= 0:
                pbar.close()

        training_time = time.time() - start
        # self.create_final_model()

        # Evaluate using the wrapper (handles inverse transform automatically)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message=".*X does not have valid feature names.*",
            )
            # y_pred = self.final_model.predict(x_test)
            y_pred = self.training_model.predict(x_test)

        self.create_metrics(y_test, y_pred, training_time)

        # return self.final_model, self.metrics
        return self.training_model, self.metrics


model_trainer = ModelTrainer()
