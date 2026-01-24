from __future__ import annotations
from typing import Dict, Any

from shared.config import config


def load_vector_schema() -> Dict[str, Any]:
    return config["prepare"]["vector_schema"]


def save_vector_schema(schema: Dict[str, Any]) -> None:
    config["prepare"]["vector_schema"] = schema
