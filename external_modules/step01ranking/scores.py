from typing import Any
from flask import jsonify
from pathlib import Path

from shared.io import atomic_write_json, load_single_entry_mapping
from .utils import get_json_path, image_root
from .cache import add


def _write_score(json_path: str, score: Any) -> tuple[bool, str | None]:
    """
    Write score to JSON metadata file.
    json_path: Full path to the .json metadata file
    """
    try:
        payload, ts, err = load_single_entry_mapping(json_path)
        if err:
            return False, f"Could not load metadata: {err}"
        if not payload:
            return False, "Invalid metadata payload"

        # Reconstruct metadata structure
        meta = {ts: payload}
        meta[ts]["score"] = score

        atomic_write_json(json_path, meta, indent=4)
        return True, None
    except Exception as e:
        return False, str(e)


def _normalize_items(data: Any) -> tuple[list[dict[str, Any]], tuple[Any, int] | None]:
    if isinstance(data, dict) and "image" in data:
        return [data], None
    if isinstance(data, dict):
        return ([{"image": k, "score": v} for k, v in data.items()], None)
    if isinstance(data, list):
        return (data, None)
    return ([], (jsonify({"error": "Invalid payload, expected list or dict"}), 400))


def submit_scores_handler(data: Any):
    items, err_resp = _normalize_items(data)
    if err_resp:
        return err_resp

    root = image_root()
    root_path = Path(root)

    result = {"ok": [], "errors": []}

    for item in items:
        if not isinstance(item, dict) or "image" not in item or "score" not in item:
            result["errors"].append(item)
            continue

        # Convert relative path (with forward slashes) to absolute path
        img_rel = item["image"].replace("\\", "/")  # Normalize to forward slashes
        img_path = root_path / img_rel
        img_abs = str(img_path)
        score = item["score"]

        json_path: str = get_json_path(img_abs)
        print(f"[submit_scores] Processing image: {img_rel}")
        print(f"[submit_scores] Absolute path: {img_abs}")
        print(f"[submit_scores] JSON path: {json_path}")
        print(f"[submit_scores] Score: {score}")

        ok, err = _write_score(json_path, score)

        # Ensure image is in cache with score, then mark as scored
        add(img_abs, score=score)  # Add to cache with score

        if ok:
            result["ok"].append(img_rel)
        else:
            result["errors"].append({"image": img_rel, "error": err})

    print(
        f"[submit_scores] Result: {len(result['ok'])} OK, {len(result['errors'])} errors"
    )
    return jsonify(result), (200 if not result["errors"] else 207)
