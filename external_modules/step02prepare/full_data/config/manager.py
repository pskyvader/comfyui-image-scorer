from typing import Dict, Any

from .....shared.config import config


def load_vector_schema() -> Dict[str, Any]:
    return config["prepare"]["vector_schema"]
