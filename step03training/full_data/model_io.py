from pathlib import Path
from typing import Any, Dict, Optional
import os
import numpy as np
import joblib
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.common.shape_calculator import calculate_linear_regressor_output_shapes
from onnxmltools.convert.lightgbm.operator_converters.LightGbm import convert_lightgbm
from lightgbm import LGBMRegressor
from sklearn.compose import TransformedTargetRegressor
from shared.paths import training_output_dir

def diagnostics_path(model_path: str) -> Path:
    path = Path(model_path)
    return path.with_suffix(".npz")


def save_model_diagnostics(model_path: str, **data: Any) -> str:
    clean_data = {k: v for k, v in data.items() if v is not None}
    
    np.savez_compressed(model_path, **clean_data)
    return model_path



def _normalize(val: Any) -> Any:
    if isinstance(val, np.ndarray):
        if val.shape == ():
            return val.item()
        return val.copy()
    return val


def load_model_diagnostics(model_path: str) -> Optional[Dict[str, Any]]:
    with np.load(Path(model_path), allow_pickle=True) as npz:
        return {k: _normalize(npz[k]) for k in npz.files}


def save_model(model: Any, model_path: str, additional_data: Dict[str, Any] = None) -> None:
    """Save a trained model to disk.

    Saves an ONNX representation to `<model_path>.onnx`.
    """
    os.makedirs(training_output_dir, exist_ok=True)
        
    if additional_data:
        save_model_diagnostics(model_path, **additional_data)

    # Save full Python object (preserves TransformedTargetRegressor wrapper)
    joblib_path = model_path + ".joblib"
    joblib.dump(model, joblib_path)
    print(f"Saved joblib model to: {joblib_path}")

    onnx_path = model_path + ".onnx"

    if isinstance(model, TransformedTargetRegressor):
        print("Warning: TransformedTargetRegressor detected. Exporting inner regressor to ONNX. Output will be in transformed space.")
        model = model.regressor_

    update_registered_converter(
        LGBMRegressor,
        "LightGbmLGBMRegressor",
        calculate_linear_regressor_output_shapes,
        convert_lightgbm,
    )

    n_inputs = getattr(model, "n_features_in_", None)
    if n_inputs is None:
        raise RuntimeError("Could not infer input dimension for ONNX export")
    initial_type = [("input", FloatTensorType([None, int(n_inputs)]))]
    onnx_model = convert_sklearn(
        model, initial_types=initial_type, target_opset={"ai.onnx.ml": 3, "": 21}
    )
    with open(onnx_path, "wb") as fh:
        fh.write(onnx_model.SerializeToString())
    print(f"Saved ONNX model to: {onnx_path}")
