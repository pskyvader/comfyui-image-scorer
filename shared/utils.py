"""Small shared utilities used across the project.

The helpers here are intentionally lightweight and thoroughly covered by unit
tests. They are written to be defensive and to avoid raising for bad input where
possible.
"""

from typing import Any
import ast
import time

from .logger import get_logger, ModuleLogger
logger: ModuleLogger = get_logger(__name__)


def parse_custom_text(val: Any) -> dict[str, Any]:
    _start = time.perf_counter()
    result: dict[str, Any]
    if val is None:
        result = {}
    elif isinstance(val, dict):
        result = val
    elif isinstance(val, str) and val:
        result = ast.literal_eval(val)
    else:
        result = {}

    return result


def first_present(d: dict[str, Any], keys: tuple[str, ...], default: Any) -> Any:
    _start = time.perf_counter()
    result: Any
    for k in keys:
        if k in d and d[k] is not None:
            result = d[k]

            break
    else:
        result = default

    return result
