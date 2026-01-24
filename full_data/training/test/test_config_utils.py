import pytest
import os
import sys
from typing import Dict, Any

# Ensure we can import from training module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from training.config_utils import around, generate_random_config, crossover_config, generate_fastest_setup, grid_base

def test_around():
    # Test valid key
    res = around("learning_rate", 0.1)
    assert len(res) > 0
    for val in res:
        assert isinstance(val, float)
        assert grid_base["learning_rate"]["min"] <= val <= grid_base["learning_rate"]["max"]

    # Test int type
    res = around("n_estimators", 100)
    for val in res:
        assert isinstance(val, int)
    
    # Test boundary conditions
    res = around("learning_rate", grid_base["learning_rate"]["min"])
    assert len(res) > 0
    res = around("learning_rate", grid_base["learning_rate"]["max"])
    assert len(res) > 0

    # Test error
    with pytest.raises(KeyError):
        around("invalid_key", 10)
        
    with pytest.raises(ValueError):
        around("learning_rate", None)


def test_generate_random_config():
    cfg = generate_random_config()
    for key in grid_base:
        assert key in cfg
        assert grid_base[key]["min"] <= cfg[key] <= grid_base[key]["max"]


def test_crossover_config():
    cfg1 = generate_random_config()
    cfg2 = generate_random_config()
    
    child = crossover_config(cfg1, cfg2)
    for key in grid_base:
        assert key in child
        assert child[key] == cfg1[key] or child[key] == cfg2[key]
    
    assert child["best_score"] == -1000000.0
    assert child["training_time"] == 0.0


def test_generate_fastest_setup():
    cfg = generate_fastest_setup()
    assert cfg["training_time"] == 99999.0
    
    # Check specific logic
    fast_params_max = {
        "min_child_samples",
        "reg_alpha",
        "reg_lambda",
        "min_split_gain",
        "learning_rate",
    }
    
    for key, cell in grid_base.items():
        if key in fast_params_max:
            # Should be max
            val = cell["max"]
            if cell["type"] == "int":
                assert cfg[key] == int(val)
            else:
                assert cfg[key] == float(val)
        else:
            # Should be min
            val = cell["min"]
            if cell["type"] == "int":
                assert cfg[key] == int(val)
            else:
                assert cfg[key] == float(val)

