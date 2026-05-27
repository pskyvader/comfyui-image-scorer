import os
import json
from shutil import move
from typing import Iterator, Any, cast

try:
    from ..config import schema as schema_module  # type: ignore[attr-defined,misc]
except ImportError:
    schema_module = None

try:
    from ....shared.io import load_json
import logging
import time
logger = logging.getLogger(__name__)
except ImportError:
    load_json = None  # type: ignore[misc]


def collect_files(str) -> Iterator[tuple[str, str]]:
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                full = os.path.join(dirpath, f)
                json_path = full.rsplit(".", 1)[0] + ".json"
                if os.path.exists(json_path):
                    yield (full, json_path)
