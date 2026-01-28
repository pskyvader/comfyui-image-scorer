from pathlib import Path
from typing import Any, Dict, Optional
import os
import numpy as np
from shared.paths import training_output_dir


def _normalize(val: Any) -> Any:
    if isinstance(val, np.ndarray):
        if val.shape == ():
            return val.item()
        return val.copy()
    return val


def load_model_diagnostics(model_path: str) -> Optional[Dict[str, Any]]:
    with np.load(Path(model_path), allow_pickle=True) as npz:
        return {k: _normalize(npz[k]) for k in npz.files}


def load_model(model_path: str) -> Any:
    """Load the trained model from .npz file.
    
    Args:
        model_path: Path to the .npz file containing the model
        
    Returns:
        The trained model object
        
    Raises:
        KeyError: If '__model_b64__' key is not found in the .npz file
    """
    import pickle
    import base64
    
    with np.load(Path(model_path), allow_pickle=True) as npz:
        if "__model_b64__" not in npz.files:
            raise KeyError(f"No '__model_b64__' key found in {model_path}. Available keys: {list(npz.files)}")
        model_b64 = npz["__model_b64__"].item()
        model_bytes = base64.b64decode(model_b64.encode('ascii'))
        return pickle.loads(model_bytes)


def save_model(model: Any, model_path: str, additional_data: Dict[str, Any] = None) -> None:
    """Save a trained model to disk.
    
    Saves both the model and diagnostic data into a single .npz file.
    Encodes the model as base64 string to work with npz format.
    """
    import pickle
    import base64
    
    os.makedirs(training_output_dir, exist_ok=True)
    
    # Pickle and encode the model to base64 (so it can be stored in npz)
    model_bytes = pickle.dumps(model)
    model_b64 = base64.b64encode(model_bytes).decode('ascii')
    
    # Create save data with encoded model
    save_data = {"__model_b64__": model_b64}
    if additional_data:
        save_data.update(additional_data)
    
    # Save everything to a single .npz file
    clean_data = {k: v for k, v in save_data.items() if v is not None}
    np.savez_compressed(model_path, **clean_data)
    print(f"Saved model and diagnostics to: {model_path}")
