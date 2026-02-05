from typing import Any, Dict, List, Union, Sequence, Set
import random

# step is relative percentage for float/int types
grid_base: Dict[str, Any] = {
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
        "random": 0.01
    },
    "max_depth": {
        # Purpose: Maximum depth of a tree. Limits the complexity of the model.
        # Speed: Deeper trees take longer to build.
        "type": "int", 
        "min": 1, 
        "max": 120, 
        "step": 0.1, 
        "random": 0.01
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
        "max": 10.0,
        "step": 0.1,
        "random": 0.01,
    },
    "reg_lambda": {
        # Purpose: L2 regularization term on weights. Penalizes large weights to reduce overfitting.
        # Speed: No significant impact on speed.
        # Starting at zero: A good first non-zero step is around 0.1 or 0.01.
        "type": "float",
        "min": 0.0,
        "max": 10.0,
        "step": 0.1,
        "random": 0.01,
    },
    "subsample": {
        # Purpose: Fraction of data samples used for each iteration (tree).
        # Speed: Lower values speed up training since less data is processed per iteration.
        "type": "float",
        "min": 0.1,
        "max": 1.0,
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
        "min": 0.0,
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
    if random.random() < cell["random"]:
        # Mutation: replace v with random value
        if cell["type"] == "int":
            v = int(random.randint(int(vmin), int(vmax)))
        else:
            v = float(random.uniform(vmin, vmax))

    result: List[Union[int, float]] = []
    candidates: Set[Union[int, float]] = set()
    if cell["type"] == "int":
        higher=int(higher)
        lower=int(lower)
        v=int(v)
        if v==lower and v>vmin:
            lower-=1
        
        if v==higher and v<vmax:
            higher+=1
            
        # Uniqueness: Use set to dedup, then sort
        candidates = {(higher), (v), (lower)}
        #candidates = {(higher), (lower)}
    if cell["type"] == "float":
        candidates = {float(higher), float(v), float(lower)}
        #candidates = {float(higher), float(lower)}

    result = sorted(list(candidates), reverse=True)
    # Final check
    result = [x for x in result if vmin <= x <= vmax]

    if not result:
        result = [v]

    return result


def generate_random_config() -> Dict[str, Any]:
    # Initialize metadata
    cfg = {"best_score": -1000000.0, "training_time": 0.0}
    for key, cell in grid_base.items():
        vmin, vmax = cell["min"], cell["max"]
        if cell["type"] == "int":
            cfg[key] = int(random.choice((int(vmin), int(vmax))))
        else:
            cfg[key] = float(random.choice((vmin, vmax)))

    return cfg


def crossover_config(cfg1: Dict[str, Any], cfg2: Dict[str, Any]) -> Dict[str, Any]:
    # Reset metadata for new offspring
    new_cfg = {"best_score": -1000000.0, "training_time": 0.0}
    for key in grid_base.keys():
        val1 = cfg1[key]
        val2 = cfg2[key]

        if random.random() < 0.5:
            new_cfg[key] = val1
        else:
            new_cfg[key] = val2

    return new_cfg


def generate_fastest_setup() -> Dict[str, Any]:
    """Generates a config likely to be fast (fewer estimators, shallow trees)."""
    # Metadata for optimizer: high training time so real runs replace it
    cfg = {"best_score": -1000000.0, "training_time": 99999.0}

    # Parameters where higher values = faster training (more regularization, larger child samples, higher LR)
    force_max = {
        "min_child_samples",
        "reg_alpha",
        "reg_lambda",
        "min_split_gain",
        "learning_rate",
    }

    for key, cell in grid_base.items():
        use_max = key in force_max
        bound_key = "max" if use_max else "min"

        if cell["type"] == "int":
            cfg[key] = int(cell[bound_key])
        else:
            cfg[key] = float(cell[bound_key])

    return cfg


def generate_slowest_setup() -> Dict[str, Any]:
    """Generates a config likely to be slow (max estimators, deep trees)."""
    # Metadata for optimizer: low score so real runs replace it
    cfg = {"best_score": -1000000.0, "training_time": 99999.0}

    # Parameters where lower values = slower training (less regularization, smaller child samples, lower LR)
    force_min = {
        "min_child_samples",
        "reg_alpha",
        "reg_lambda",
        "min_split_gain",
        "learning_rate",
    }

    for key, cell in grid_base.items():
        use_min = key in force_min
        bound_key = "min" if use_min else "max"

        if cell["type"] == "int":
            cfg[key] = int(cell[bound_key])
        else:
            cfg[key] = float(cell[bound_key])

    return cfg
