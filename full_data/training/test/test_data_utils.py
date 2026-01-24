import sys
from pathlib import Path
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Mock shared.io before importing training.data_utils
# Using sys.modules to mock load_jsonl since it is imported directly
with patch('shared.io.load_jsonl') as mock_load_jsonl:
    from training import data_utils


def test_load_training_data_mismatch():
    with patch('training.data_utils.load_jsonl') as mock_load:
        mock_load.side_effect = [
            np.array([[1]]), # vectors
            np.array([1, 2]) # scores (mismatch)
        ]
        
        with pytest.raises(RuntimeError, match='Mismatched vector and score counts'):
            data_utils.load_training_data("vectors.jsonl", "scores.jsonl")

def test_load_training_data_success():
    with patch('training.data_utils.load_jsonl') as mock_load:
        mock_load.side_effect = [
            np.array([[1], [2]]), # vectors
            np.array([10, 20])    # scores
        ]
        
        X, y = data_utils.load_training_data("vectors.jsonl", "scores.jsonl")
        assert len(X) == 2
        assert len(y) == 2
        assert X[0][0] == 1
        assert y[1] == 20

def test_prepare_plot_data():
    y = np.array([1.0, 2.0, np.nan, 4.0])
    preds = np.array([1.1, np.nan, 3.1, 4.1])
    
    y_out, p_out = data_utils.prepare_plot_data(y, preds)
    # expect index 0 and 3
    assert len(y_out) == 2
    assert len(p_out) == 2
    assert y_out[0] == 1.0
    assert p_out[0] == 1.1
    assert y_out[1] == 4.0
    assert p_out[1] == 4.1

def test_prepare_plot_data_empty():
    y = np.array([np.nan])
    preds = np.array([1.0])
    y_out, p_out = data_utils.prepare_plot_data(y, preds)
    assert len(y_out) == 0
    assert len(p_out) == 0
