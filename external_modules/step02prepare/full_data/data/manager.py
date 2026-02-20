import os
import json
import sys
from shutil import move
from typing import Iterator, Tuple, List, Dict, Any, cast,Optional


from external_modules.step02prepare.full_data.config import schema as schema_module
from shared.config import config
from shared.io import load_index_list, load_json

from transformers import (
    AutoModel,
    AutoProcessor,
)

import torch

from PIL import Image as PILImage
from tqdm import tqdm


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


def encode_images_to_vectors(paths: List[str], batch_size: int) -> List[List[float]]:
    load_clip()

    vectors: List[List[float]] = []
    total = len(paths)
    processed = 0

    pbar = (
        tqdm(total=total, desc="Encoded", unit="img", file=sys.stdout) if tqdm else None
    )
    try:
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i : i + batch_size]
            images = []
            for p in batch_paths:
                img = PILImage.open(p).convert("RGB")
                images.append(img)
            inputs = _clip_processor(images=images, return_tensors="pt")
            inputs = {k: v.to(_clip_model.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = _clip_model.get_image_features(**inputs)
            batch_vecs = outputs.cpu().numpy().astype(float).tolist()
            for vec in batch_vecs:
                if len(vec) != schema_module.IMAGE_VEC_LEN:
                    msg = f"CLIP returned unexpected vector length {len(vec)}, expected {schema_module.IMAGE_VEC_LEN}"
                    raise RuntimeError(msg)
            vectors.extend(batch_vecs)
            processed += len(batch_vecs)
            if pbar is not None:
                pbar.update(len(batch_vecs))
            else:
                print(f"Encoded {processed}/{total} images...")
    finally:
        if pbar is not None:
            pbar.close()
    return vectors


def collect_files(root: str) -> Iterator[Tuple[str, str]]:
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                full = os.path.join(dirpath, f)
                json_path = full.rsplit(".", 1)[0] + ".json"
                if os.path.exists(json_path):
                    yield (full, json_path)


def load_index(path: str) -> List[str]:
    data, err = load_index_list(path, [])
    if err:
        print(f"Error loading index {path}: {err}")
        return []
    return [str(item) for item in data]


def save_index(idx: List[str], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(list(idx), f, indent=4)

def pad_existing_vectors(
    vectors_file: str,
    schema: Dict[str, Any],
    prompt_representation: Optional[Dict[str, Any]] = None,
) -> None:
    if not os.path.exists(vectors_file):
        return

    # Backwards-compatible default for prompt_representation when not provided in tests
    if prompt_representation is None:
        prompt_representation = {"mode": "embedding", "dim": 768}

    feature_len = schema_module.get_feature_vector_length(
        schema["slots"],
        prompt_representation["mode"],
        int(prompt_representation["dim"]),
    )
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
