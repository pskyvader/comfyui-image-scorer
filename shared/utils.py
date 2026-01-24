"""Small shared utilities used across the project.

The helpers here are intentionally lightweight and thoroughly covered by unit
tests. They are written to be defensive and to avoid raising for bad input where
possible.
"""

from typing import Any, Dict, List, Tuple, Union
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


def one_hot(index: int, length: int) -> List[int]:
    """Return a one-hot encoded vector of ``length`` with 1 at ``index``.

    Out-of-range indexes produce a zero vector rather than raising.
    """
    v = [0] * length
    if 0 <= index < length:
        v[index] = 1
    return v


def binary_presence(indices: List[int], length: int) -> List[int]:
    """Return a binary presence vector with 1s at the supplied indices.

    Indices outside [0, length) are ignored.
    """
    v = [0] * length
    for i in indices:
        if 0 <= i < length:
            v[i] = 1
    return v


def weighted_presence(*args: Any) -> List[float]:
    """Build a weighted presence vector.

    Supported call signatures:
    - weighted_presence(indexed_terms: List[Tuple[int, float]], length: int)
      where ``indexed_terms`` is a list of (index, weight).
    - weighted_presence(indices: List[int], weights: List[float], length: int)

    Values outside the valid range are ignored. If multiple weights are
    provided for the same index, the maximum weight is kept.
    """
    if len(args) == 2 and isinstance(args[0], list) and isinstance(args[1], int):
        indexed_terms, length = args
    elif len(args) == 3 and isinstance(args[0], list) and isinstance(args[1], list) and isinstance(args[2], int):
        indices, weights, length = args
        indexed_terms = list(zip(indices, weights))
    else:
        raise TypeError('Invalid arguments for weighted_presence')
    v = [0.0] * length
    for idx, weight in indexed_terms:
        if 0 <= idx < length:
            v[idx] = max(v[idx], float(weight))
    return v
