import os
import json
from shutil import move
from typing import Iterator, Tuple, Any, cast


from ..config import schema as schema_module
from .....shared.config import config
from .....shared.io import load_json

from transformers import (
    AutoModel,
    AutoProcessor,
)




_clip_model: Any = None
_clip_processor: Any = None


def load_clip() -> None:
    global _clip_model, _clip_processor
    if _clip_model is not None:
        return
    device = config["vision_model"]["device"]
    VISION_MODEL_ID = config["vision_model"]["name"]
    if device != "cuda":
        raise RuntimeError("`clip_device` not set to 'cuda'")

    try:
        print(f"Loading Vision Model: {VISION_MODEL_ID}...")
        model = AutoModel.from_pretrained(VISION_MODEL_ID, local_files_only=True)
        processor = AutoProcessor.from_pretrained(
            VISION_MODEL_ID, local_files_only=True, use_fast=True
        )
    except Exception:
        # Fallback to online if not in cache (and if internet is available)
        print("Vision model not found in cache, attempting download...")
        model = AutoModel.from_pretrained(VISION_MODEL_ID)
        processor = AutoProcessor.from_pretrained(VISION_MODEL_ID)
    model = model.eval()
    model.to(device)
    _clip_model = model
    _clip_processor = processor
    print(f"Vision model loaded on device: {device} ")





def collect_files(root: str) -> Iterator[Tuple[str, str]]:
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                full = os.path.join(dirpath, f)
                json_path = full.rsplit(".", 1)[0] + ".json"
                if os.path.exists(json_path):
                    yield (full, json_path)





    expected_len = schema_module.IMAGE_VEC_LEN + feature_len
    data, err = load_json(vectors_file, expect=list, default=[])
    if err:
        return
    changed = False
    data_typed: list[Any] = cast(list[Any], data or [])
    for i, vec in enumerate(data_typed):
        if not isinstance(vec, list):
            continue
        vec_typed: list[Any] = cast(list[Any], vec)
        vec_len = len(vec_typed)
        if vec_len < expected_len:
            needed = expected_len - vec_len
            data_typed[i] = vec_typed + [0.0] * needed
            changed = True
    if changed:
        tmp_path = vectors_file + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as outf:
            json.dump(data_typed, outf)
        move(tmp_path, vectors_file)
