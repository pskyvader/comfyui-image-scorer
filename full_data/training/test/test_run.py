import sys
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Patch lgb and onnx before importing run to avoid import errors if not installed or to speed up
with patch('lightgbm.LGBMRegressor'), patch('onnxruntime.InferenceSession'):
    from training import run

def test_around():
    # Test int with no mutation (mock random.random to return 1.0)
    # run.grid_base["n_estimators"] -> step=0.1. 100 * 1.1 = 110, 100 * 0.9 = 90
    with patch('random.random', return_value=1.0):
        res = run.around("n_estimators", 100)
        assert isinstance(res, list)
        assert 100 in res
        
        # Test float
        res = run.around("learning_rate", 0.05)
        assert isinstance(res, list)

def test_around_mutation():
    # Test mutation (mock random.random to return 0.0 which is < step)
    with patch('random.random', return_value=0.0):
        # mock randint to return a specific value
        with patch('random.randint', return_value=500): # 500 is within [50, 1000] and different from 100
             res = run.around("n_estimators", 100)
             # The new logic calculate neighbors around the mutated value.
             # 100 -> mutated to 500.
             # step is 0.1
             # lower = 500 * 0.9 = 450
             # higher = 500 * 1.1 = 550
             # result = sorted([550, 500, 450]) -> [550, 500, 450]
             assert 500 in res
             assert len(res) == 3
        
        # mock uniform for float
        with patch('random.uniform', return_value=0.2): # Within [0.001, 0.5]
             res = run.around("learning_rate", 0.01)
             # 0.01 -> mutated to 0.2
             # step 0.1
             # lower = 0.18
             # higher = 0.22
             assert 0.2 in res
             assert len(res) == 3

def test_generate_combos():
    grid = {
        "a": [1, 2],
        "b": [3, 4]
    }
    combos = run.generate_combos(grid, max_combos=10)
    assert len(combos) == 4
    
    combos_limited = run.generate_combos(grid, max_combos=2)
    assert len(combos_limited) == 2

def test_prepare_optimization_setup():
    base_cfg = {
        "learning_rate": 0.01,
        "n_estimators": 100,
        "num_leaves": 31,
        "max_depth": 10,
        "min_child_samples": 20,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "min_split_gain": 0.0,
        "early_stopping_rounds": 10,
    }
    # Mock resolve_path and config in run
    # Note: run.resolve_path is imported from helpers.
    with patch('training.run.config', {"training": {"output_dir": "test/out"}}), \
         patch('training.run.resolve_path', return_value="/abs/test/out"), \
         patch('os.makedirs'):
        
        grid, temp = run.prepare_optimization_setup(base_cfg)
        assert "learning_rate" in grid
        assert os.path.normpath(temp) == os.path.normpath(os.path.join("/abs/test/out", "temp_model"))

def test_prepare_optimization_setup_missing_key():
    base_cfg = {}
    with patch('training.run.config', {"training": {"output_dir": "test/out"}}), \
         patch('training.run.resolve_path', return_value="/abs/test/out"), \
         patch('os.makedirs'):
        
        with pytest.raises(ValueError):
            run.prepare_optimization_setup(base_cfg)
