from typing import Any, Dict, Tuple, cast, List, Optional

import os
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
    f1_score,
    precision_score,
    recall_score,
)

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import lightgbm as lgb

from shared.config import config
from shared.training.plot import plot_metric, LivePlotCallback

from external_modules.step03training.full_data.config_utils import grid_base

from shared.paths import models_dir


class ModelTrainer:

    METRIC_DIRECTIONS: Dict[str, Dict[str, bool]] = {
        "lambdarank": {
            "ndcg": True,  # higher is better
        },
        "multiclass": {
            "multi_logloss": False,  # lower is better
            "multi_error": False,
        },
        "binary": {
            "binary_logloss": False,
            "auc": True,
        },
        "regression": {
            "l2": False,
            "rmse": False,
            "l1": False,
            "r2": True,
        },
    }

    def __init__(self) -> None:
        self.training_model = None
        self.eval_metrics: List[Any] = []
        # self.final_model = None
        self.n_estimators = None
        self.test_size = None
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
        if "test_size" in params:
            self.test_size = float(params.pop("test_size"))
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

    def create_callbacks(self, progress_bar: Any, enable_plotting: bool = False) -> None:
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

        self.plot_callback = None
        if self.user_verbosity >= 0:
            # Use tqdm progress bar
            def progress_bar_callback(_):
                progress_bar.update(1)

            callbacks.append(progress_bar_callback)
            # Add plotting callback only if enabled (not during HPO temporary evaluations)
            if enable_plotting:
                graph_path = os.path.join(models_dir, "graph.png")
                self.plot_callback = LivePlotCallback(save_path=graph_path, frequency=30)
                callbacks.append(self.plot_callback)

        self.callbacks = callbacks

    def create_metrics(
        self, y_test: np.ndarray, y_pred: np.ndarray, training_time: float
    ) -> None:
        if not self.training_model:
            raise ModuleNotFoundError("training model not found")

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
        objective = model.get_params()["objective"]

        if objective in {"regression", "regression_l2", "l2"}:
            # Flatten arrays for regression (predictions are 1D)
            y_pred_flat = np.asarray(y_pred).ravel()
            y_test_flat = np.asarray(y_test).ravel()

            mae = float(mean_absolute_error(y_test_flat, y_pred_flat))
            rmse = float(root_mean_squared_error(y_test_flat, y_pred_flat))
            r2 = float(r2_score(y_test_flat, y_pred_flat))

            self.result_metrics.update(
                {
                    "mae": mae,
                    "rmse": rmse,
                    "r2": r2,
                }
            )

        elif objective in {"multiclass", "binary"}:
            # Classification metrics
            y_pred_flat = np.asarray(y_pred)
            if y_pred_flat.ndim > 1:
                if objective == "binary":
                    predictions = (y_pred_flat[:, 1] > 0.5).astype(int)
                else:
                    predictions = np.argmax(y_pred_flat, axis=1)
            else:
                predictions = (
                    (y_pred_flat > 0.5).astype(int)
                    if objective == "binary"
                    else y_pred_flat
                )

            acc = float(accuracy_score(y_test, predictions))
            self.result_metrics["accuracy"] = acc

            # F1-scores (better for imbalanced datasets)
            try:
                macro_f1 = float(
                    f1_score(y_test, predictions, average="macro", zero_division=0)
                )
                weighted_f1 = float(
                    f1_score(y_test, predictions, average="weighted", zero_division=0)
                )
                macro_precision = float(
                    precision_score(
                        y_test, predictions, average="macro", zero_division=0
                    )
                )
                macro_recall = float(
                    recall_score(y_test, predictions, average="macro", zero_division=0)
                )
                self.result_metrics["macro_f1"] = macro_f1
                self.result_metrics["weighted_f1"] = weighted_f1
                self.result_metrics["macro_precision"] = macro_precision
                self.result_metrics["macro_recall"] = macro_recall
            except Exception:
                pass

            # log loss only if probabilities available
            try:
                ll = float(log_loss(y_test, y_pred))
                self.result_metrics["log_loss"] = ll
            except Exception:
                pass

        # Ranking usually uses internal metrics only

        # Extract LightGBM evaluation results
        evaluation_results = getattr(model, "evals_result_", None)
        metric_dict = None
        if evaluation_results:
            target_set = "valid" if "valid" in evaluation_results else None
            if not target_set:
                print(f"Warning: not valid set found, trying training...")
                target_set = "train" if "train" in evaluation_results else None
            if target_set:
                metric_dict = evaluation_results[target_set]

        primary_metric = self.eval_metrics[0]
        if metric_dict and evaluation_results:
            self.result_metrics["curves"] = {}

            for dataset_name, metrics_dict in evaluation_results.items():
                self.result_metrics["curves"][dataset_name] = {}

                for metric_name, values in metrics_dict.items():
                    self.result_metrics["curves"][dataset_name][metric_name] = values

            # Define main score = first metric in eval_metric list
            if primary_metric in metric_dict:
                self.result_metrics["score"] = metric_dict[primary_metric][-1]
                self.result_metrics["primary_metric"] = primary_metric

    def train_model(
        self,
        config_dict: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        enable_plotting: bool = False,
    ) -> Tuple[Any, Dict[str, Any]]:
        if config["training"]["device"] != "cuda":
            raise ValueError("Training must use CUDA device.")

        # test_size = float(config["training"]["test_size"])
        # test_size=config_dict.pop("test_size")
        self.create_training_model(config_dict)

        random_state = config["training"]["random_state"]
        split_res = cast(
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
            tuple(
                train_test_split(
                    X, y, test_size=self.test_size, random_state=random_state
                )
            ),
        )
        x_train, x_test, y_train, y_test = split_res

        progress_bar = tqdm(total=self.n_estimators, desc="Training LightGBM")
        self.create_callbacks(progress_bar, enable_plotting=enable_plotting)
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
                progress_bar.close()
                # Plot final training results only once
                #if self.plot_callback is not None:
                #    self.plot_callback.plot_final_results()

        training_time = time.time() - start

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message=".*X does not have valid feature names.*",
            )
            y_pred = self.training_model.predict(x_test)

        self.create_metrics(y_test, y_pred, training_time)

        return self.training_model, self.result_metrics


model_trainer = ModelTrainer()
