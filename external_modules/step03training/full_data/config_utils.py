from typing import Any
import random
from shared.training.model_trainer import grid_base


def generate_random_config() -> dict[str, Any]:
    # Initialize metadata
    cfg = {"best_score": -1000000.0, "training_time": 0.0}
    for key, cell in grid_base.items():
        vmin, vmax = cell["min"], cell["max"]

        # if cell["type"] == "int":
        #     cfg[key] = int(random.choice((int(vmin), int(vmax))))
        # else:
        #     cfg[key] = float(random.choice((vmin, vmax)))

        if cell["type"] == "int":
            cfg[key] = int(random.randint(int(vmin), int(vmax)))
        else:
            cfg[key] = float(random.uniform(vmin, vmax))

    return cfg


def crossover_config(cfg1: dict[str, Any], cfg2: dict[str, Any]) -> dict[str, Any]:
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


def generate_fastest_setup() -> dict[str, Any]:
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


def generate_slowest_setup() -> dict[str, Any]:
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
