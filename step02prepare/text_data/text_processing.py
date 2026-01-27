import json
import os
from typing import Dict, List, Set, Tuple, TextIO, Any
from PIL import Image

from step02prepare.full_data.data.metadata import load_metadata_entry
from step02prepare.full_data.features.meta import parse_entry_meta

from step02prepare.full_data.features.terms import extract_terms

from shared.config import config
from shared.io import load_index_set, atomic_write_json
from shared.paths import image_root


def extract_text_components(entry: Dict[str, Any]) -> Dict[str, Any]:
    normalization = config["prepare"]["normalization"]
    (
        cfg,
        steps,
        lora_weight,
        sampler,
        scheduler,
        model,
        lora,
        _,
        _,
        _,
    ) = parse_entry_meta(entry, normalization)
    pos_terms = extract_terms(
        entry.get("positive_prompt", ""),
        {"and", "or", "with", "a", "an", "the", "in", "is", "at", "to", "by", "of"},
        {"girl", ",", "years"},
    )
    neg_terms = extract_terms(
        entry.get("negative_prompt", ""),
        {"and", "or", "with", "a", "an", "the", "in", "is", "at", "to", "by", "of"},
        {"girl", ","},
    )
    score = None
    raw_score = entry.get("score")
    if raw_score is not None:
        try:
            score = float(raw_score)
        except (ValueError, TypeError):
            pass
    return {
        "negative_prompt": entry.get("negative_prompt"),
        "positive_terms": pos_terms,
        "negative_terms": neg_terms,
        "sampler": sampler,
        "scheduler": scheduler,
        "model": model,
        "lora": lora,
        "cfg": cfg,
        "steps": steps,
        "lora_weight": lora_weight,
        "score": score,
    }


def process_text_files(
    files: List[Tuple[str, str]],
    processed_files: Set[str],
    outf: TextIO,
    error_log: List[Dict[str, str]],
) -> int:
    """Extract text metadata for scored images and append to the export file."""
    new_count = 0
    for img_path, meta_path in files:
        file_id = os.path.relpath(img_path, image_root).replace("\\", "/")
        if file_id in processed_files:
            continue
        entry, ts, err = load_metadata_entry(meta_path)
        if entry is None:
            if err:
                error_log.append({"file": file_id, "reason": err})
            continue
        if "score" not in entry:
            continue

        width, height = 0, 0
        aspect_ratio = 0.0
        try:
            with Image.open(img_path) as img:
                width, height = img.size
                if height > 0:
                    aspect_ratio = round(width / height, 4)
        except Exception as e:
            error_log.append({"file": file_id, "reason": f"image_read_error: {e}"})

        # Inject dimensions into entry before calling extract_text_components
        # extract_text_components calls parse_entry_meta which requires width/height in entry
        entry["width"] = width
        entry["height"] = height
        entry["aspect_ratio"] = aspect_ratio

        text_data = extract_text_components(entry)

        text_data["file_id"] = file_id
        text_data["timestamp"] = ts
        text_data["width"] = width
        text_data["height"] = height
        text_data["aspect_ratio"] = aspect_ratio

        outf.write(json.dumps(text_data) + "\n")
        processed_files.add(file_id)
        new_count += 1
    return new_count


def load_text_index(path: str) -> Set[str]:
    data, err = load_index_set(path, [])
    if err and err != "not_found":
        print(f"Error loading text index {path}: {err}")
    return data


def save_text_index(index: Set[str], path: str) -> None:
    atomic_write_json(path, sorted(index), indent=2)
