"""Score submission handlers for the ranking UI.

This module provides helpers to normalize incoming score payloads and to
update the corresponding per-image JSON metadata files. The handlers are
written to be easily unit-testable (file I/O is isolated behind small helpers
so it can be patched in tests).
"""

from typing import Any, Dict, Tuple, List
from flask import jsonify
    
from shared.io import atomic_write_json
from ranking.utils import get_json_path, load_meta


def _write_score(json_path: str, score: Any) -> Tuple[bool, str | None]:
    """Write `score` into the JSON file at ``json_path``.

    Returns (True, None) on success or (False, error_message) on failure.
    """
    meta, timestamp, err = load_meta(json_path)
    if err:
        return (False, err)
    if meta is None or timestamp is None:
        return (False, 'Invalid metadata contents')
    meta[timestamp]['score'] = score
    try:
        atomic_write_json(json_path, meta, indent=4)
    except Exception as exc:  # pragma: no cover - IO failure
        return (False, f'Failed to write JSON: {exc}')
    return (True, None)

def _normalize_items(data: Any) -> Tuple[List[Dict[str, Any]], Tuple[Any, int] | None]:
    if isinstance(data, dict) and 'image' in data:
        return [data], None
    if isinstance(data, dict):
        return ([{'image': k, 'score': v} for k, v in data.items()], None)
    if isinstance(data, list):
        return (data, None)
    return ([], (jsonify({'error': 'Invalid payload, expected list or dict'}), 400))


def submit_scores_handler(data: Any) -> Tuple[Any, int] | Any:
    items, err_resp = _normalize_items(data)
    if err_resp:
        return err_resp

    results = {'ok': [], 'errors': []}
    for item in items:
        if not isinstance(item, dict) or 'image' not in item or 'score' not in item:
            results['errors'].append({'item': item, 'error': 'Invalid item format'})
            continue
        img_path = item['image']
        score = item['score']
        json_path: str = get_json_path(img_path)
        ok, err = _write_score(json_path, score)
        if ok:
            results['ok'].append(img_path)
        else:
            results['errors'].append({'image': img_path, 'error': err, 'json_path': json_path})
    status: int = 200 if not results['errors'] else 207
    return jsonify(results), status