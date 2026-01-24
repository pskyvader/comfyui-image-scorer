import os
from pathlib import Path
from typing import Any, List, Tuple, Dict
from flask import send_from_directory
from shared.io import load_single_entry_mapping

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def serve_file(subpath: str) -> Any:
    directory = os.path.dirname(subpath) or '.'
    filename = os.path.basename(subpath)
    return send_from_directory(directory, filename, conditional=True)


def get_json_path(img_path: str) -> str:
    base: str = os.path.splitext(img_path)[0]
    return base + '.json'


def find_images(root_dir: str) -> List[str]:
    files: List[str] = []
    for path, _, filenames in os.walk(root_dir):
        for name in filenames:
            if Path(name).suffix.lower() in IMAGE_EXTENSIONS:
                full_path = os.path.join(path, name)
                files.append(full_path.replace("\\", "/"))
    return sorted(files)


def load_meta(json_path: str) -> Tuple[Dict[str, Any] | None, str | None, str | None]:
    payload, ts, err = load_single_entry_mapping(json_path)
    if err:
        if err == 'not_found':
            return (None, None, 'JSON file missing')
        if err.startswith('load_error'):
            return (None, None, f'Failed to read JSON: {err}')
        return (None, None, err)
    return ({ts: payload}, ts, None)


def load_metadata(image_path: str) -> Any | None:
    metadata_path = get_json_path(image_path)
    payload, ts, err = load_single_entry_mapping(metadata_path)
    if err:
        return None
    return {ts: payload}


def get_unscored_images(root: str) -> List[str]:
    images = find_images(root)
    unscored: List[str] = []
    for img in images:
        meta = load_metadata(img)
        if not meta:
            continue
        latest_entry = next(iter(meta.values()), None)
        if not latest_entry:
            continue
        if not isinstance(latest_entry, dict):
            print(
                f"Invalid metadata format for image {img}", "type", type(latest_entry)
            )
            continue
        if "score" in latest_entry:
            continue
        unscored.append(img)
    return unscored
