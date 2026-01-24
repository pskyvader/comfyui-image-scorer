import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training import model_io


def test_save_and_load_model_diagnostics(tmp_path):
    model_base = tmp_path / "model"
    diag_path = model_io.save_model_diagnostics(str(model_base), metrics={"r2": 0.9}, loss_curve=np.array([1.0, 0.5]))

    assert diag_path.exists()
    loaded = model_io.load_model_diagnostics(str(model_base))
    assert loaded is not None
    assert "metrics" in loaded and "loss_curve" in loaded
    assert loaded["metrics"]["r2"] == 0.9
    assert np.allclose(np.asarray(loaded["loss_curve"], dtype=float), np.array([1.0, 0.5]))


def test_load_model_diagnostics_missing(tmp_path):
    missing_base = tmp_path / "missing"
    assert model_io.load_model_diagnostics(str(missing_base)) is None
