import os
from typing import Any, Dict, List, Tuple
from full_data.prepare.features.meta import parse_entry_meta
from full_data.prepare.features.terms import extract_terms
from shared.config import  config
from shared.io import atomic_write_json, load_single_entry_mapping


def write_error_log(error_log: List[Dict[str, str]], path: str) -> None:
    if error_log:
        atomic_write_json(path, error_log, indent=2)
    elif os.path.exists(path):
        os.remove(path)


def load_metadata_entry(meta_path: str) -> Tuple[Dict[str, Any] | None, str | None, str | None]:
    payload, ts, err = load_single_entry_mapping(meta_path)
    if err:
        if err == "not_found":
            return (None, None, "json_not_found")
        if err.startswith("load_error"):
            print(f"Error loading {meta_path}: {err}")
            return (None, None, "bad_json")
        return (None, None, err)
    return (payload, ts, None)


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
        width_norm,
        height_norm,
        ar_norm,
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
