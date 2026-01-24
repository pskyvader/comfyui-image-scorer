from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any, Tuple, TypeVar, Callable

T = TypeVar("T")


def _recursive_parse_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _recursive_parse_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_recursive_parse_json(v) for v in obj]
    elif isinstance(obj, str):
        s = obj.strip()
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            try:
                parsed = json.loads(obj)
                if isinstance(parsed, (dict, list)):
                    return _recursive_parse_json(parsed)
            except Exception:
                pass
    return obj


def load_json(
    path: str,
    expect: type | tuple[type, ...] | None,
    default: T | None,
) -> Tuple[T | Any, str | None]:
    if not path:
        return default, "missing_path"
    path_obj = Path(path)
    if not path_obj.exists():
        return default, "not_found"
    try:
        with path_obj.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
            data = _recursive_parse_json(data)
    except Exception as exc:  # pragma: no cover - io errors are environment specific
        return default, f"load_error:{exc}"
    if expect is not None and not isinstance(data, expect):
        return default, "invalid_type"
    return data, None


def atomic_write_json(path: str, data: Any, *, indent: int | None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp_path = f"{path}.tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=indent)
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except Exception:
                pass
        os.replace(tmp_path, path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def load_json_list(path: str, default_list: list[Any]) -> Tuple[list[Any], str | None]:
    data, err = load_json(path, expect=list, default=default_list)
    return (data or [], err)


def load_json_set(path: str, default_list: list[str]) -> Tuple[set[str], str | None]:
    data, err = load_json(path, expect=list, default=default_list)
    try:
        return set(data or []), err
    except Exception:
        return set(), err or "invalid_type"


def load_single_entry_mapping(
    path: str,
) -> Tuple[dict[str, Any] | None, str | None, str | None]:
    data, err = load_json(path, expect=dict, default=None)
    if err:
        return None, None, err
    if data is None or len(data.keys()) != 1:
        return None, None, "invalid_keys"
    key = next(iter(data.keys()))
    payload = data[key]
    if not isinstance(payload, dict):
        return None, None, "invalid_payload"
    return (payload, str(key), None)


def load_or_default(path: str, factory: Callable[[], T]) -> T:
    data, err = load_json(path, expect=None, default=None)
    if err:
        return factory()
    return data if data is not None else factory()


def load_index_list(path: str, fallback: list[Any]) -> Tuple[list[Any], str | None]:
    """Load a list-based index with standardized error codes."""
    data, err = load_json(path, expect=list, default=fallback)
    if err == "not_found":
        return [], None
    return data or [], err


def load_index_set(path: str, fallback: list[str]) -> Tuple[set[str], str | None]:
    """Load a set of string identifiers with consistent error handling."""
    data, err = load_json(path, expect=list, default=fallback)
    if err == "not_found":
        return set(), None
    if err:
        return set(), err
    try:
        return set(data or []), None
    except Exception:
        return set(), "invalid_type"


def load_json_list_robust(path: str) -> list[Any]:
    data, err = load_json_list(path, [])
    if err and os.path.exists(path):
        try:
            corrupt_path = f"{path}.corrupt"
            i = 1
            while os.path.exists(corrupt_path):
                corrupt_path = f"{path}.corrupt-{i}"
                i += 1
            os.replace(path, corrupt_path)
            print(f"Error loading {path}: {err}. Moved corrupt file to {corrupt_path}")
        except Exception as e2:
            print(f"Error loading {path}: {err}; additionally failed to move corrupt file: {e2}")
    return data

from typing import List, cast
import numpy as np

VECTOR_KEYS: Tuple[str, ...] = ('vector', 'features', 'vec', 'embedding')
SCORE_KEYS: Tuple[str, ...] = ('score', 'value', 'y')


def is_numeric(obj: Any) -> bool:
    return isinstance(obj, (int, float)) and not isinstance(obj, bool)


def coerce_numeric_list(obj: Any) -> List[float] | None:
    if not isinstance(obj, list):
        return None
    if not all(is_numeric(v) for v in obj):
        return None
    return [float(cast(float, v)) for v in obj]


def extract_vectors(obj: Any, vector_keys: Tuple[str, ...]) -> List[List[float]]:
    vectors: List[List[float]] = []
    direct = coerce_numeric_list(obj)
    if direct is not None:
        vectors.append(direct)
        return vectors
    if isinstance(obj, list):
        for el in obj:
            candidate = coerce_numeric_list(el)
            if candidate is not None:
                vectors.append(candidate)
                continue
            if isinstance(el, dict):
                el_dict = cast(dict[str, Any], el)
                for key in vector_keys:
                    candidate = coerce_numeric_list(el_dict[key] if key in el_dict else None)
                    if candidate is not None:
                        vectors.append(candidate)
                        break
        return vectors
    if isinstance(obj, dict):
        obj_dict = cast(dict[str, Any], obj)
        for key in vector_keys:
            candidate = coerce_numeric_list(obj_dict[key] if key in obj_dict else None)
            if candidate is not None:
                vectors.append(candidate)
                break
        if not vectors:
            for val in obj_dict.values():
                candidate = coerce_numeric_list(val)
                if candidate is not None:
                    vectors.append(candidate)
                    break
    return vectors


def extract_scores(obj: Any, score_keys: Tuple[str, ...]) -> List[float]:
    scores: List[float] = []
    if is_numeric(obj):
        scores.append(float(cast(float, obj)))
        return scores
    direct = coerce_numeric_list(obj)
    if direct is not None:
        if len(direct) == 1:
            scores.append(direct[0])
        elif direct:
            scores.append(float(sum(direct) / len(direct)))
        return scores
    if isinstance(obj, list):
        for el in obj:
            if is_numeric(el):
                scores.append(float(cast(float, el)))
                continue
            if isinstance(el, dict):
                el_dict = cast(dict[str, Any], el)
                for key in score_keys:
                    if key in el_dict:
                        value = el_dict[key]
                        if is_numeric(value):
                            scores.append(float(cast(float, value)))
                    else: 
                        # This happens in JSON parsing where structure is loose.
                        # We cannot crash here as we are scanning for potential keys.
                        pass
                        break
        return scores
    if isinstance(obj, dict):
        obj_dict = cast(dict[str, Any], obj)
        for key in score_keys:
            value = obj_dict[key] if key in obj_dict else None
            if is_numeric(value):
                scores.append(float(cast(float, value)))
                break
    return scores


def process_json_obj(obj: Any, kind: str, vector_keys: Tuple[str, ...], score_keys: Tuple[str, ...]) -> List[Any]:
    if kind == 'vector':
        return extract_vectors(obj, vector_keys)
    return extract_scores(obj, score_keys)


def flatten_top_level(obj: Any, kind: str, vector_keys: Tuple[str, ...], score_keys: Tuple[str, ...]) -> List[Any]:
    if kind == 'score':
        direct_scores = coerce_numeric_list(obj)
        if direct_scores is not None:
            return [float(v) for v in direct_scores]
    if kind == 'vector':
        direct_vector = coerce_numeric_list(obj)
        if direct_vector is not None:
            return [direct_vector]

    if isinstance(obj, list):
        entries: List[Any] = []
        if kind == 'vector' and obj:
            homogeneous_vectors = coerce_numeric_list(obj[0]) is not None
            if homogeneous_vectors:
                for el in obj:
                    candidate = coerce_numeric_list(el)
                    if candidate is not None:
                        entries.append(candidate)
                return entries
        for el in obj:
            entries.extend(process_json_obj(el, kind, vector_keys, score_keys))
        return entries

    return process_json_obj(obj, kind, vector_keys, score_keys)


def load_jsonl(
    path: str,
    kind: str = 'vector',
    vector_keys: Tuple[str, ...] = VECTOR_KEYS,
    score_keys: Tuple[str, ...] = SCORE_KEYS,
) -> np.ndarray:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f'Missing required data file: {file_path}')

    raw_content = file_path.read_text(encoding='utf-8').strip()
    if not raw_content:
        raise ValueError(f'File is empty: {file_path}')

    try:
        parsed: Any = json.loads(raw_content)
        entries = flatten_top_level(parsed, kind, vector_keys, score_keys)
    except json.JSONDecodeError:
        # Fallback for Newline Delimited JSON (.jsonl)
        entries = []
        for line in raw_content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                line_obj = json.loads(line)
                # Assume each line is a single object containing data
                extracted = process_json_obj(line_obj, kind, vector_keys, score_keys)
                entries.extend(extracted)
            except json.JSONDecodeError:
                continue

    if not entries:
        raise ValueError(f'No valid {kind} entries found in {file_path}')

    arr = np.asarray(entries, dtype=float)
    if kind == 'vector':
        if arr.ndim != 2:
            raise ValueError(f'Expected 2D vectors in {file_path}; got shape {arr.shape}')
        if arr.size == 0:
            raise ValueError(f'No vector data present in {file_path}')
        return arr

    arr = arr.reshape(-1)
    if arr.size == 0:
        raise ValueError(f'No score data present in {file_path}')
    return arr

