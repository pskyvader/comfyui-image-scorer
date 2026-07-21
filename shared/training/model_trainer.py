from typing import Any, Union, Sequence, cast
import random

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

from sklearn.model_selection import train_test_split
import lightgbm as lgb

from ..logger import get_logger
from ..config import config
from ..io import load_single_jsonl
from ..loaders.training_loader import training_loader
from ..paths import index_file
from .calibration import build_score_calibration
from .pair_data import build_pairwise_dataset, load_comparison_records

logger = get_logger(__name__)


# step is relative percentage for float/int types
grid_base: dict[str, Any] = {
    "learning_rate": {
        # Purpose: Shrinks the contribution of each tree by learning_rate. Controls how fast the model learns.
        # Speed: Lower values slow down training significantly as more trees (n_estimators) are needed to reach convergence.
        "type": "float",
        "min": 0.001,
        "max": 0.5,
        "step": 0.1,
        "random": 0.01,
    },
    "n_estimators": {
        # Purpose: Number of boosting iterations (trees) to fit.
        # Speed: Training time increases linearly with n_estimators.
        "type": "int",
        "min": 10,
        "max": 2000,
        "step": 0.1,
        "random": 0.01,
    },
    "num_leaves": {
        # Purpose: Maximum number of leaves in one tree. Main parameter to control model complexity.
        # Speed: Higher values decrease training speed and increase memory usage.
        "type": "int",
        "min": 2,
        "max": 1024,
        "step": 0.1,
        "random": 0.01,
    },
    "max_depth": {
        # Purpose: Maximum depth of a tree. Limits the complexity of the model.
        # Speed: Deeper trees take longer to build.
        "type": "int",
        "min": 1,
        "max": 120,
        "step": 0.1,
        "random": 0.01,
    },
    "min_child_samples": {
        # Purpose: Minimum number of data points needed in a leaf. Helps prevent overfitting.
        # Speed: no significant impact.
        "type": "int",
        "min": 1,
        "max": 400,
        "step": 0.1,
        "random": 0.01,
    },
    "reg_alpha": {
        # Purpose: L1 regularization term on weights. Increases sparsity (sets some weights to exactly zero).
        # Speed: No significant impact on speed.
        # Starting at zero: A good first non-zero step is around 0.1 or 0.01.
        "type": "float",
        "min": 0.0,
        "max": 20.0,
        "step": 0.1,
        "random": 0.01,
    },
    "reg_lambda": {
        # Purpose: L2 regularization term on weights. Penalizes large weights to reduce overfitting.
        # Speed: No significant impact on speed.
        # Starting at zero: A good first non-zero step is around 0.1 or 0.01.
        "type": "float",
        "min": 0.0,
        "max": 20.0,
        "step": 0.1,
        "random": 0.01,
    },
    "subsample": {
        # Purpose: Fraction of data samples used for each iteration (tree).
        # Speed: Lower values speed up training since less data is processed per iteration.
        "type": "float",
        "min": 0.1,
        "max": 0.99,
        "step": 0.1,
        "random": 0.01,
    },
    "colsample_bytree": {
        # Purpose: Fraction of features (columns) randomly selected for each iteration (tree). It uses a different random subset of columns for each tree.
        # Speed: Lower values speed up training significantly if there are many features.
        "type": "float",
        "min": 0.1,
        "max": 1.0,
        "step": 0.1,
        "random": 0.01,
    },
    "min_split_gain": {
        # Purpose: Minimum loss reduction required to make a further partition on a leaf node.
        # Speed: Can improve speed by pruning the tree early (similar to max_depth).
        # Starting at zero: A good first non-zero step is around 0.1 or 0.01.
        "type": "float",
        "min": 0.05,
        "max": 0.5,
        "step": 0.1,
        "random": 0.01,
    },
    "early_stopping_rounds": {
        # Purpose: Stops training if the validation score doesn't improve for this many rounds.
        # Speed: Can drastically reduce training time by stopping early when convergence is reached.
        "type": "int",
        "min": 5,
        "max": 200,
        "step": 0.1,
        "random": 0.01,
    },
    "test_size": {
        # Purpose: Minimum loss reduction required to make a further partition on a leaf node.
        # Speed: Can improve speed by pruning the tree early (similar to max_depth).
        # Starting at zero: A good first non-zero step is around 0.1 or 0.01.
        "type": "float",
        "min": 0.001,
        "max": 0.99,
        "step": 0.1,
        "random": 0.01,
    },
}


def around(label: str, val: Union[int, float, None]) -> Sequence[Union[int, float]]:
    cell = grid_base[label]
    if cell["type"] not in ("int", "float"):
        raise ValueError(f"Unsupported type for grid search cell: {cell['type']}")
    if val is None:
        raise ValueError(f"Value for grid search cell '{label}' is None")

    # Check base value is within limits and of correct type
    vmin, vmax = cell["min"], cell["max"]
    if cell["type"] == "int":
        v = int(max(vmin, min(vmax, val)))
    else:
        v = float(max(vmin, min(vmax, val)))

    lower = max(v * (1 - cell["step"]), vmin + cell["step"] / 10 if vmin == 0 else vmin)
    higher = min(v * (1 + cell["step"]), vmax)

    # Random mutation based on probability
    # if random.random() < cell["random"]:
    # Mutation: replace v with random value
    if cell["type"] == "int":
        v = int(random.randint(int(vmin), int(vmax)))
    else:
        v = float(random.uniform(vmin, vmax))

    result: list[Union[int, float]] = []
    candidates: set[Union[int, float]] = set()
    if cell["type"] == "int":
        higher: int = int(higher)
        lower: int = int(lower)

        v = int(v)
        if v == lower and v > vmin:
            lower -= 1

        if v == higher and v < vmax:
            higher += 1

        # Uniqueness: Use set to dedup, then sort
        candidates = {(higher), (v), (lower)}
        # candidates = {(higher), (lower)}
    if cell["type"] == "float":
        # candidates = {float(higher), float(v), float(lower)}
        candidates = {float(higher), float(lower)}
        v_float = v
        v = int(v)
        if v == lower and v > vmin:
            lower -= 1

        if v == higher and v < vmax:
            higher += 1

        # Uniqueness: Use set to dedup, then sort
        candidates = {(higher), (v_float), (lower)}
        # candidates = {(higher), (lower)}
    result = sorted(list(candidates), reverse=False)
    # Final check
    result = [x for x in result if vmin <= x <= vmax]

    if not result:
        result = [v]

    return result


class ModelTrainer:

    METRIC_DIRECTIONS: dict[str, dict[str, bool]] = {
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
            "r2": True,
            "l2": False,
            "rmse": False,
            "l1": False,
        },
    }

    def __init__(self) -> None:
        self.training_model = None
        self.eval_metrics: list[Any] = []
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
    ) -> tuple[str, float, bool]:
        """Custom R2 metric for LightGBM evaluation."""
        return ("r2", float(r2_score(y_true, y_pred)), True)

    @staticmethod
    def _pairwise_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
        """Compute pairwise accuracy for flattened 2-row comparison groups."""

        y_true_flat = np.asarray(y_true).ravel()
        y_pred_flat = np.asarray(y_pred).ravel()
        if len(y_true_flat) < 2 or len(y_pred_flat) < 2:
            return None

        total = 0
        correct = 0
        for i in range(0, min(len(y_true_flat), len(y_pred_flat)) - 1, 2):
            left_true = float(y_true_flat[i])
            right_true = float(y_true_flat[i + 1])
            if left_true == right_true:
                continue

            total += 1
            left_pred = float(y_pred_flat[i])
            right_pred = float(y_pred_flat[i + 1])
            if left_true > right_true:
                correct += int(left_pred > right_pred)
            else:
                correct += int(right_pred > left_pred)

        if total == 0:
            return None

        return float(correct / total)

    def _build_score_calibration(self, X: np.ndarray) -> dict[str, Any] | None:
        """Build a fixed raw->score calibration for lambdarank models."""

        if self.training_model is None:
            return None

        objective = self.training_model.get_params().get("objective")
        if objective != "lambdarank":
            return None

        try:
            target_scores = training_loader.load_scores_array()
        except Exception as exc:
            logger.warning("Skipping score calibration: failed to load scores: %s", exc)
            return None

        if len(target_scores) != len(X):
            logger.warning(
                "Skipping score calibration: scores length (%d) does not match X length (%d).",
                len(target_scores),
                len(X),
            )
            return None

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message=".*X does not have valid feature names.*",
            )
            raw_scores = self.training_model.predict(X)

        calibration = build_score_calibration(raw_scores, target_scores)
        if calibration is None:
            return None

        return calibration

    def create_training_model(self, config_dict: dict[str, Any]):

        # Base LightGBM parameters
        params: dict[str, Any] = {
            "random_state": config["training"]["random_state"],
            "verbosity": -1,
            "objective": config["training"]["objective"],
            "device_type": "gpu",
            "importance_type": "gain",
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
            self.eval_metrics = ["r2", self.r2_metric, "l2", "rmse", "l1"]

    def create_callbacks(
        self, progress_bar: Any, status_bar: Any, enable_plotting: bool = False
    ) -> None:
        # Setup callbacks for logging
        callbacks: list[Any] = []
        # Always suppress default logger to ensure clean output
        callbacks.append(lgb.log_evaluation(period=-1))

        if self.early_stopping_rounds is not None:
            callbacks.append(
                lgb.early_stopping(
                    stopping_rounds=self.early_stopping_rounds, verbose=False
                )
            )

        if self.user_verbosity >= 0:

            def pbar_callback(_):
                progress_bar.update(1)

            callbacks.append(pbar_callback)

        # Note: Custom callbacks with closures (like progress_bar_callback) can't be pickled,
        # causing issues in LightGBM. Only use built-in LightGBM callbacks.
        self.plot_callback = None

        self.callbacks = callbacks

    def create_metrics(
        self, y_test: np.ndarray, y_pred: np.ndarray, training_time: float
    ) -> None:
        if not self.training_model:
            raise ModuleNotFoundError("training model not found")

        model = self.training_model
        model_type = type(model).__name__

        self.result_metrics: dict[str, Any] = {
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
                    "r2": r2,
                    "mae": mae,
                    "rmse": rmse,
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
                    f1_score(y_test, predictions, average="macro", zero_division="0")
                )
                weighted_f1 = float(
                    f1_score(y_test, predictions, average="weighted", zero_division="0")
                )
                macro_precision = float(
                    precision_score(
                        y_test, predictions, average="macro", zero_division="0"
                    )
                )
                macro_recall = float(
                    recall_score(
                        y_test, predictions, average="macro", zero_division="0"
                    )
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

        elif objective == "lambdarank":
            pairwise_accuracy = self._pairwise_accuracy(y_test, y_pred)
            if pairwise_accuracy is not None:
                self.result_metrics["pairwise_accuracy"] = pairwise_accuracy

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
            # print("metric and evaluation present", flush=True)
            # print(
            #     f"primary_metric: {primary_metric}, type: {type(primary_metric)}",
            #     flush=True,
            # )
            # # print(f"execution: {primary_metric()}", flush=True)
            # print("metric_dict.keys()", metric_dict.keys(), flush=True)
            # print(f"evaluation_results.keys: {evaluation_results.keys()}", flush=True)
            self.result_metrics["curves"] = {}

            for dataset_name, metrics_dict in evaluation_results.items():
                # print("dataset_name:", dataset_name, flush=True)
                self.result_metrics["curves"][dataset_name] = {}

                for metric_name, values in metrics_dict.items():
                    # print("metric_name:", metric_name, flush=True)
                    self.result_metrics["curves"][dataset_name][metric_name] = values

            # Define main score = first metric in eval_metric list
            if primary_metric in metric_dict:
                # print("primary metric present", primary_metric, flush=True)
                self.result_metrics["score"] = metric_dict[primary_metric][-1]
                self.result_metrics["primary_metric"] = primary_metric

        if objective == "lambdarank" and "score" not in self.result_metrics:
            fallback_accuracy = self._pairwise_accuracy(y_test, y_pred)
            if fallback_accuracy is not None:
                self.result_metrics["score"] = fallback_accuracy
                self.result_metrics["primary_metric"] = "pairwise_accuracy"

    def train_model_pairs(
        self,
        config_dict: dict[str, Any],
        X: np.ndarray,
        comparisons: list[dict[str, Any]],
        index_list: list[str],
        enable_plotting: bool = False,
    ) -> tuple[Any, dict[str, Any]]:
        if config["training"]["device"] != "cuda":
            raise ValueError("Training must use CUDA device.")

        self.create_training_model(config_dict)
        if not isinstance(self.training_model, lgb.LGBMRanker):
            raise TypeError("Pairwise training requires a lambdarank objective.")

        if len(comparisons) < 2:
            raise ValueError("Need at least two comparisons for pairwise training.")

        random_state = config["training"]["random_state"]
        comparison_indices = list(range(len(comparisons)))
        split_res = cast(
            tuple[list[int], list[int]],
            tuple(
                train_test_split(
                    comparison_indices,
                    test_size=self.test_size,
                    random_state=random_state,
                )
            ),
        )
        train_indices, test_indices = split_res
        train_comparisons = [comparisons[i] for i in train_indices]
        test_comparisons = [comparisons[i] for i in test_indices]
        logger.debug("building pairwise dataset...")

        (
            x_train,
            y_train,
            group_train,
            weight_train,
            valid_train_pairs,
        ) = build_pairwise_dataset(X, index_list, train_comparisons)
        (
            x_test,
            y_test,
            group_test,
            _weight_test,
            valid_test_pairs,
        ) = build_pairwise_dataset(X, index_list, test_comparisons)

        if valid_train_pairs == 0 or valid_test_pairs == 0:
            raise RuntimeError(
                "Pairwise training split produced an empty train or validation set."
            )

        progress_bar = tqdm(
            total=self.n_estimators, desc="Training LightGBM", position=0, delay=3.0
        )
        status_bar = tqdm(total=0, desc="Status", position=1, bar_format="{desc}", delay=3.0)
        logger.debug("creating callbacks...")

        self.create_callbacks(
            progress_bar, enable_plotting=enable_plotting, status_bar=status_bar
        )
        start = time.time()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message=".*X does not have valid feature names.*",
            )

            parameters: dict[str, Any] = {
                "X": x_train,
                "y": y_train,
                "sample_weight": weight_train,
                "eval_set": [(x_train, y_train), (x_test, y_test)],
                "eval_names": ["train", "valid"],
                "eval_metric": self.eval_metrics,
                "callbacks": self.callbacks,
            }

            if self.training_model is None:
                raise RuntimeError("Training model was not created")
            logger.debug("beginning training...")
            self.training_model.fit(
                **parameters,
                group=group_train,
                eval_group=[group_train, group_test],
            )

            if self.training_model is None:
                raise RuntimeError("Training failed")

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

        self.create_metrics(y_test, np.asarray(y_pred), training_time)
        score_calibration = self._build_score_calibration(X)
        if score_calibration is not None:
            self.result_metrics["score_calibration"] = score_calibration
        self.result_metrics["pairwise_pairs_train"] = valid_train_pairs
        self.result_metrics["pairwise_pairs_valid"] = valid_test_pairs

        return self.training_model, self.result_metrics

    def train_model(
        self,
        config_dict: dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        enable_plotting: bool = False,
    ) -> tuple[Any, dict[str, Any]]:
        if config["training"]["device"] != "cuda":
            raise ValueError("Training must use CUDA device.")

        objective = config["training"]["objective"]
        if objective == "lambdarank":
            comparison_rows = load_comparison_records()
            raw_index_list = load_single_jsonl(index_file)
            index_list = [str(item) for item in raw_index_list]
            if not comparison_rows:
                raise FileNotFoundError(
                    "No comparison data found. Run prepare data to export comparisons.jsonl first."
                )
            if not index_list:
                raise FileNotFoundError(
                    "No index data found. Run prepare data to export index.jsonl first."
                )
            return self.train_model_pairs(
                config_dict=config_dict,
                X=X,
                comparisons=comparison_rows,
                index_list=index_list,
                enable_plotting=enable_plotting,
            )

        # test_size = float(config["training"]["test_size"])
        # test_size=config_dict.pop("test_size")
        self.create_training_model(config_dict)

        random_state = config["training"]["random_state"]
        split_res = cast(
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
            tuple(
                train_test_split(
                    X, y, test_size=self.test_size, random_state=random_state
                )
            ),
        )
        x_train, x_test, y_train, y_test = split_res

        progress_bar = tqdm(
            total=self.n_estimators, desc="Training LightGBM", position=0, delay=3.0
        )
        status_bar = tqdm(total=0, desc="Status", position=1, bar_format="{desc}", delay=3.0)
        self.create_callbacks(
            progress_bar, enable_plotting=enable_plotting, status_bar=status_bar
        )
        start = time.time()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",

                category=UserWarning,
                message=".*X does not have valid feature names.*",
            )

            parameters: dict[str, Any] = {
                "X": x_train,
                "y": y_train,
                "eval_set": [(x_train, y_train), (x_test, y_test)],
                "eval_names": ["train", "valid"],
                "eval_metric": self.eval_metrics,
                "callbacks": self.callbacks,
            }

            if self.training_model is None:
                raise RuntimeError("Training model was not created")

            if isinstance(self.training_model, lgb.LGBMRanker):
                self.training_model.fit(
                    **parameters,
                    group=[len(y_train)],
                    eval_group=[[len(y_train)], [len(y_test)]],
                )
            else:
                self.training_model.fit(**parameters)

            if self.training_model is None:
                raise RuntimeError("Training failed")

            if self.user_verbosity >= 0:
                progress_bar.close()
                # Plot final training results only once
                # if self.plot_callback is not None:
                #    self.plot_callback.plot_final_results()

        training_time = time.time() - start

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message=".*X does not have valid feature names.*",
            )
            y_pred = self.training_model.predict(x_test)

        # print(f"y_test: {y_test[100:]}")
        # print(f"y_pred: {y_pred[100:]}")

        self.create_metrics(y_test, np.asarray(y_pred), training_time)

        return self.training_model, self.result_metrics


model_trainer = ModelTrainer()
