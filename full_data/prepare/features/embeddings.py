from __future__ import annotations
from typing import Optional
import numpy as np

from sentence_transformers import SentenceTransformer

_MODEL: Optional[object] = None
_MODEL_NAME: Optional[str] = None


def load_model(name: str, dim: int, device: str) -> object:
    global _MODEL, _MODEL_NAME
    if _MODEL is None or _MODEL_NAME != name:
        try:
            _MODEL = SentenceTransformer(name, device=device)
        except Exception:
            # Try with default device if cuda fails or explicit device issue
            _MODEL = SentenceTransformer(name)
        _MODEL_NAME = name
    
    if _MODEL.get_sentence_embedding_dimension() != dim:
        raise ValueError(
            f"Model dimensions mismatch: config says {dim}, model {name} has {_MODEL.get_sentence_embedding_dimension()}"
        )
            
    return _MODEL


def encode_prompt(prompt: str, embedding_model: object) -> np.ndarray:
    emb = embedding_model.encode(prompt)
    return np.asarray(emb, dtype=np.float32)
