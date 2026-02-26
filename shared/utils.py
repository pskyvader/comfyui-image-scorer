"""Small shared utilities used across the project.

The helpers here are intentionally lightweight and thoroughly covered by unit
tests. They are written to be defensive and to avoid raising for bad input where
possible.
"""

from typing import Any, Dict, Tuple
import ast


def parse_custom_text(val: Any = None) -> Dict[str, Any]:
    """Parse a stored custom text value into a dictionary.

    Accepts None, an already-parsed dict, or a string containing a Python
    literal representation (e.g., "{'a':1}"). Returns an empty dict for any
    invalid input.
    """
    if val is None:
        return {}
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        if not val:
            return {}
        try:
            return ast.literal_eval(val)
        except Exception:
            return {}
    return {}


def first_present(d: Dict[str, Any], keys: Tuple[str, ...], default: Any = None) -> Any:
    """Return the first key from ``keys`` present in ``d`` with a non-None value.

    If none of the keys are present, return ``default``.
    """
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default
