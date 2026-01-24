from typing import Any, Dict, List, Tuple, Union
import ast

def parse_custom_text(val: Any = None) -> Dict[str, Any]:
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
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default

def one_hot(index: int, length: int) -> List[int]:
    v = [0] * length
    if 0 <= index < length:
        v[index] = 1
    return v

def binary_presence(indices: List[int], length: int) -> List[int]:
    v = [0] * length
    for i in indices:
        if 0 <= i < length:
            v[i] = 1
    return v

def weighted_presence(*args: Any) -> List[float]:
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
