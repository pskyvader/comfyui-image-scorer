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
import lightgbm as lgb

from shared.config import config

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

    def create_callbacks(self, progress_bar: Any) -> None:
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
            def progress_bar_callback(_):
                progress_bar.update(1)

            callbacks.append(progress_bar_callback)
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

    def create_metrics(
        self, y_test, y_pred, training_time: float
    ):
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
        if metric_dict:
            self.result_metrics["curves"] = {}

            for dataset_name, metrics_dict in evaluation_results.items():
                self.result_metrics["curves"][dataset_name] = {}

                for metric_name, values in metrics_dict.items():
                    self.result_metrics["curves"][dataset_name][metric_name] = values

            # Define main score = first metric in eval_metric list
            if primary_metric in metric_dict:
                self.result_metrics["score"] = metric_dict[primary_metric][-1]

    def train_model(
        self,
        config_dict: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[Any, Dict[str, Any]]:
        if not self.training_model:
            self.create_training_model(config_dict)

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

        progress_bar = tqdm(total=self.n_estimators, desc="Training LightGBM")
        self.create_callbacks(progress_bar)
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
