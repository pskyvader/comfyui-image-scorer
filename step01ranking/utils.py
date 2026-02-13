import os
from pathlib import Path
from typing import Any, List, Dict
from flask import send_from_directory, abort
from urllib.parse import unquote
from shared.io import load_single_entry_mapping
from step01ranking.cache import (
    add,
    get,
    in_cache,
    disable,
    fast_serve,
    set_absolute_total,
)
from shared.config import PROJECT_ROOT, config
import random

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
_image_root=None
_image_list_cache=[]

def serve_file(subpath: str) -> Any:
    """Serve file relative to image_root, properly decoded."""
    root = Path(image_root())
    path = root / unquote(subpath)
    if not path.exists() or not path.is_file():
        abort(404)
    return send_from_directory(str(path.parent), path.name, conditional=True)


def get_json_path(img_path: str) -> str:
    return os.path.splitext(img_path)[0] + ".json"


def load_metadata(image_path: str) -> Dict[str, Any] | None:
    metadata_path = get_json_path(image_path)
    payload, ts, err = load_single_entry_mapping(metadata_path)
    if err:
        return None
    return {ts: payload}


def image_root() -> str:
    global _image_root
    if _image_root:
        return _image_root
    img_root = config["image_root"]
    root_path = Path(img_root)
    if not root_path.is_absolute():
        root_path = PROJECT_ROOT.joinpath(root_path).resolve()
    _image_root=str(root_path)
    return _image_root


def find_images(root_dir: str) -> List[str]:
    files: List[str] = []
    for path, _, filenames in os.walk(root_dir):
        for name in filenames:
            if Path(name).suffix.lower() in IMAGE_EXTENSIONS:
                #files.append(os.path.join(path, name).replace("\\", "/"))
                full_path = os.path.join(path, name)
                files.append(full_path.replace("\\", "/"))

    
    
    random.shuffle(files)
    set_absolute_total(len(files))
    return files


def scan_batch(root: str, limit: int = 100) -> bool:
    # if fast_serve():
    #     return False

    all_images = find_images(root)
    added_any_unscored = False
    batch_count = 0

    for img in all_images:
        if img in _image_list_cache or in_cache(img):
            continue
        if img not in _image_list_cache:
            _image_list_cache.append(img)
        
        add(img)

        meta = load_metadata(img)
        entry = next(iter(meta.values()), {}) if meta else {}
        if not entry or not isinstance(entry, dict):
            disable(img)
            continue
        
        if "score" in entry:
            disable(img)
            continue

        added_any_unscored = True
        batch_count += 1

        if batch_count >= limit:
            break

    return added_any_unscored


def get_unscored_images(root: str) -> List[str]:
    """
    Return **truly unscored images** for the client.
    Paths are relative to image_root and use forward slashes.
    """
    cached = get(valid_only=True)  # cache already only valid images
    root_path = Path(root)
    unscored: List[str] = []

    for img in cached:
        #meta = load_metadata(img)
        #entry = next(iter(meta.values()), {}) if meta else {}
        #if entry and isinstance(entry, dict) and "score" not in entry:
            # relative path for Flask + forward slashes
        try:
            rel_path = str(Path(img).relative_to(root_path))#.replace("\\", "/")
        except ValueError:
            rel_path = os.path.basename(img)
        unscored.append(rel_path)
    #print(cached[0],unscored[0])
    return unscored
