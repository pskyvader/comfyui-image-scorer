from __future__ import annotations

import time
from typing import Any

from shared.logger import get_logger
from shared.utils import first_present, parse_custom_text

logger = get_logger(__name__)


def parse_entry_meta(entry: dict[str, Any], normalization: dict[str, Any]) -> tuple[
    float,
    float,
    float,
    Any,
    Any,
    Any,
    Any,
    float,
    float,
    float,
    float,
    float,
    float,
]:
    _start = time.perf_counter()
    custom = parse_custom_text(entry["custom_text"]) if "custom_text" in entry else {}

    cfg_val = float(
        first_present(
            custom if "cfg" in custom else entry,
            ("cfg", "cfg_scale", "guidance_scale", "cfgScale"),
            0.0,
        )
    )
    steps_val = float(
        first_present(
            custom if "steps" in custom else entry,
            ("steps", "step", "sampling_steps", "sampler_steps", "num_inference_steps"),
            0.0,
        )
    )

    cfg_norm = max(0.0, min(1.0, cfg_val / float(normalization["cfg_max"])))
    steps_norm = max(0.0, min(1.0, steps_val / float(normalization["steps_max"])))

    w_max = (
        float(normalization["width_max"]) if "width_max" in normalization else 4096.0
    )
    h_max = (
        float(normalization["height_max"]) if "height_max" in normalization else w_max
    )
    ar_max = (
        float(normalization["aspect_ratio_max"])
        if "aspect_ratio_max" in normalization
        else 5.0
    )

    width_val = (
        float(entry["width"]) if "width" in entry else float(entry["final_width"])
    )
    height_val = (
        float(entry["height"]) if "height" in entry else float(entry["final_height"])
    )
    ar_val = (
        float(entry["aspect_ratio"])
        if "aspect_ratio" in entry
        else (width_val / height_val if height_val > 0 else 1.0)
    )

    original_width_norm = max(0.0, min(1.0, width_val / w_max))
    original_height_norm = max(0.0, min(1.0, height_val / h_max))
    original_ar_norm = max(0.0, min(1.0, ar_val / ar_max))

    final_width_val = float(entry["final_width"])
    final_height_val = float(entry["final_height"])
    final_ar_val = (
        float(entry["final_aspect_ratio"])
        if "final_aspect_ratio" in entry
        else (final_width_val / final_height_val if final_height_val > 0 else 1.0)
    )

    final_width_norm = max(0.0, min(1.0, final_width_val / w_max))
    final_height_norm = max(0.0, min(1.0, final_height_val / h_max))
    final_ar_norm = max(0.0, min(1.0, final_ar_val / ar_max))

    lw_raw = first_present(
        custom if "lora_weight" in custom else entry,
        ("lora_weight", "lora_strength", "lora"),
        None,
    )
    lora_weight = 0.0 if lw_raw is None else float(lw_raw)
    lora_weight = max(0.0, min(1.0, lora_weight))

    sampler_name = first_present(
        custom if "sampler" in custom else entry,
        ("sampler", "sampler_name"),
        None,
    )
    scheduler_name = first_present(
        custom if "scheduler" in custom else entry,
        ("scheduler", "scheduler_name"),
        None,
    )
    model_name = first_present(
        custom if "model" in custom else entry,
        ("model", "model_name", "base_model", "checkpoint", "ckpt"),
        None,
    )
    lora_name = first_present(
        custom if "lora" in custom else entry,
        ("lora", "lora_name"),
        None,
    )

    result = (
        cfg_norm,
        steps_norm,
        lora_weight,
        sampler_name,
        scheduler_name,
        model_name,
        lora_name,
        original_width_norm,
        original_height_norm,
        original_ar_norm,
        final_width_norm,
        final_height_norm,
        final_ar_norm,
    )
    return result
