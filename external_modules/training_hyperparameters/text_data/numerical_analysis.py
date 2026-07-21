"""Types and analysis functions for numerical analysis of text/hyperparameter data."""

from typing import Any
import numpy as np

from ....shared.vectors.terms import extract_terms

# ========================================================================
#  Types
# ========================================================================

ElementType = tuple[int, float]  # (nodeid, score)
ElementList = list[ElementType]

FloatKeyDict = dict[float, ElementList]  # {value_2dp: [(nodeid,score),...]}
IntKeyDict = dict[int, ElementList]
StrKeyDict = dict[str, ElementList]

PromptWeightMap = dict[float, IntKeyDict]  # {weight: {position: [(nodeid,score),...]}}
PromptType = dict[str, PromptWeightMap]  # {term: {weight: {position: [...]}}}

ContinuousType = dict[str, FloatKeyDict]  # {param: {value: [(nodeid,score),...]}}

PositionalType = dict[str, ContinuousType]  # {field: {component: {value: [(nodeid,score),...]}}}

FlattenedType = dict[
    str, ElementList
]  # all dict types compressed with a.b.c format, all float (2 decimal) and int parsed to str

# ========================================================================
#  Extractors
# ========================================================================


def extract_lora(
    line: dict[str, Any],
    score: float,
    nodeid: int,
    lora_data: ContinuousType,
) -> None:
    lora_name = line.pop("lora", None)
    lora_weight = line.pop("lora_weight", None)
    if lora_name is not None and lora_weight is not None:
        val = float(lora_weight)
        lora_data.setdefault(lora_name, {}).setdefault(val, []).append((nodeid, score))


def extract_prompts(
    line: dict[str, Any],
    score: float,
    nodeid: int,
    positive_prompt: PromptType,
    negative_prompt: PromptType,
) -> None:
    positive_text = line.pop("positive_prompt", None)
    if positive_text:
        try:
            for prompt, weight, idx in extract_terms(positive_text).terms:
                positive_prompt.setdefault(prompt, {}).setdefault(
                    weight, {}
                ).setdefault(idx, []).append((nodeid, score))
        except Exception as e:
            print(f"Error parsing positive prompt: {e}")

    negative_text = line.pop("negative_prompt", None)
    if negative_text:
        try:
            for prompt, weight, idx in extract_terms(negative_text).terms:
                negative_prompt.setdefault(prompt, {}).setdefault(
                    weight, {}
                ).setdefault(idx, []).append((nodeid, score))
        except Exception as e:
            print(f"Error parsing negative prompt: {e}")


AGE_LABELS = [
    "0-2",
    "3-9",
    "10-19",
    "20-29",
    "30-39",
    "40-49",
    "50-59",
    "60-69",
    "70+",
]
GENDER_LABELS = ["Female", "Male"]
RACE_LABELS = [
    "Black",
    "East Asian",
    "Indian",
    "Latino_Hispanic",
    "Middle Eastern",
    "Southeast Asian",
    "White",
]

POSE_KEY: dict[int, str] = {
    0: "nose",
    2: "left_eye",
    5: "right_eye",
    7: "left_ear",
    8: "right_ear",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
}

HAND_KEY: dict[int, str] = {
    0: "wrist",
    4: "thumb_tip",
    8: "index_tip",
    12: "middle_tip",
    16: "ring_tip",
    20: "pinky_tip",
}


def extract_face_logits(
    line: dict[str, Any],
    score: float,
    nodeid: int,
    face_age_data: ContinuousType,
    face_gender_data: ContinuousType,
    face_race_data: ContinuousType,
) -> None:
    face_logits = line.pop("face_logits", None)
    if face_logits and len(face_logits) >= 18:
        for j, lbl in enumerate(AGE_LABELS):
            val = float(face_logits[j])
            face_age_data.setdefault(lbl, {}).setdefault(val, []).append(
                (nodeid, score)
            )
        for j, lbl in enumerate(GENDER_LABELS):
            val = float(face_logits[9 + j])
            face_gender_data.setdefault(lbl, {}).setdefault(val, []).append(
                (nodeid, score)
            )
        for j, lbl in enumerate(RACE_LABELS):
            val = float(face_logits[11 + j])
            face_race_data.setdefault(lbl, {}).setdefault(val, []).append(
                (nodeid, score)
            )


def extract_face_bbox(
    line: dict[str, Any],
    score: float,
    nodeid: int,
    continuous_data: ContinuousType,
    positional_data: PositionalType,
) -> None:
    face_bbox_val = line.pop("face_bbox", None)
    if face_bbox_val and len(face_bbox_val) > 0:
        b = face_bbox_val[0]
        for comp_idx, comp_name in enumerate(["x", "y", "w", "h", "conf"]):
            if comp_idx < len(b):
                key = f"face_bbox_{comp_name}"
                val = float(b[comp_idx])
                continuous_data.setdefault(key, {}).setdefault(val, []).append(
                    (nodeid, score)
                )
        pd = positional_data.setdefault("face_bbox", {})
        for comp_idx, comp_name in enumerate(("x", "y", "w", "h")):
            if comp_idx < len(b):
                val = max(0.0, min(1.0, float(b[comp_idx])))
                pd.setdefault(comp_name, {}).setdefault(val, []).append((nodeid, score))


def extract_pose_landmarks(
    line: dict[str, Any],
    score: float,
    nodeid: int,
    pose_data: ContinuousType,
    positional_data: PositionalType,
) -> None:
    pose_raw = line.pop("body_pose", None)
    if pose_raw and len(pose_raw) > 0 and len(pose_raw[0]) >= 132:
        arr = pose_raw[0]
        for idx, name in POSE_KEY.items():
            x_key = f"pose_{name}_x"
            y_key = f"pose_{name}_y"
            vis_key = f"pose_{name}_vis"
            for target_key, val in [
                (x_key, arr[4 * idx]),
                (y_key, arr[4 * idx + 1]),
                (vis_key, arr[4 * idx + 3]),
            ]:
                fval = float(val)
                pose_data.setdefault(target_key, {}).setdefault(fval, []).append(
                    (nodeid, score)
                )
            pos_key = f"pose_{name}"
            pd = positional_data.setdefault(pos_key, {})
            x_val = max(0.0, min(1.0, float(arr[4 * idx])))
            y_val = max(0.0, min(1.0, float(arr[4 * idx + 1])))
            pd.setdefault("x", {}).setdefault(x_val, []).append((nodeid, score))
            pd.setdefault("y", {}).setdefault(y_val, []).append((nodeid, score))


def extract_hand_landmarks(
    line: dict[str, Any],
    score: float,
    nodeid: int,
    lh_data: ContinuousType,
    rh_data: ContinuousType,
    positional_data: PositionalType,
) -> None:
    lh_raw = line.pop("left_hand", None)
    if lh_raw and len(lh_raw) > 0 and len(lh_raw[0]) >= 84:
        arr = lh_raw[0]
        for idx, name in HAND_KEY.items():
            x_key = f"lh_{name}_x"
            y_key = f"lh_{name}_y"
            for target_key, val in [(x_key, arr[4 * idx]), (y_key, arr[4 * idx + 1])]:
                fval = float(val)
                lh_data.setdefault(target_key, {}).setdefault(fval, []).append(
                    (nodeid, score)
                )
            pos_key = f"lh_{name}"
            pd = positional_data.setdefault(pos_key, {})
            x_val = max(0.0, min(1.0, float(arr[4 * idx])))
            y_val = max(0.0, min(1.0, float(arr[4 * idx + 1])))
            pd.setdefault("x", {}).setdefault(x_val, []).append((nodeid, score))
            pd.setdefault("y", {}).setdefault(y_val, []).append((nodeid, score))

    rh_raw = line.pop("right_hand", None)
    if rh_raw and len(rh_raw) > 0 and len(rh_raw[0]) >= 84:
        arr = rh_raw[0]
        for idx, name in HAND_KEY.items():
            x_key = f"rh_{name}_x"
            y_key = f"rh_{name}_y"
            for target_key, val in [(x_key, arr[4 * idx]), (y_key, arr[4 * idx + 1])]:
                fval = float(val)
                rh_data.setdefault(target_key, {}).setdefault(fval, []).append(
                    (nodeid, score)
                )


def extract_image_sizes(
    line: dict[str, Any],
    score: float,
    nodeid: int,
    continuous_data: ContinuousType,
    positional_data: PositionalType,
) -> None:
    orig_w = line.pop("original_width", None)
    orig_h = line.pop("original_height", None)
    final_w = line.pop("final_width", None)
    final_h = line.pop("final_height", None)

    if orig_w is not None:
        val = float(orig_w)
        continuous_data.setdefault("original_width", {}).setdefault(val, []).append(
            (nodeid, score)
        )
    if orig_h is not None:
        val = float(orig_h)
        continuous_data.setdefault("original_height", {}).setdefault(val, []).append(
            (nodeid, score)
        )
    if final_w is not None:
        val = float(final_w)
        continuous_data.setdefault("final_width", {}).setdefault(val, []).append(
            (nodeid, score)
        )
    if final_h is not None:
        val = float(final_h)
        continuous_data.setdefault("final_height", {}).setdefault(val, []).append(
            (nodeid, score)
        )
    if orig_w is not None and orig_h is not None:
        ar = float(orig_w) / float(orig_h)
        continuous_data.setdefault("original_aspect_ratio", {}).setdefault(
            ar, []
        ).append((nodeid, score))
    if final_w is not None and final_h is not None:
        ar = float(final_w) / float(final_h)
        continuous_data.setdefault("final_aspect_ratio", {}).setdefault(ar, []).append(
            (nodeid, score)
        )
        ps = positional_data.setdefault("final_size", {})
        ps.setdefault("w", {}).setdefault(float(final_w), []).append((nodeid, score))
        ps.setdefault("h", {}).setdefault(float(final_h), []).append((nodeid, score))
    if orig_w is not None and orig_h is not None:
        pos_os = positional_data.setdefault("original_size", {})
        pos_os.setdefault("w", {}).setdefault(float(orig_w), []).append((nodeid, score))
        pos_os.setdefault("h", {}).setdefault(float(orig_h), []).append((nodeid, score))


def extract_remaining_fields(
    line: dict[str, Any],
    score: float,
    nodeid: int,
    discrete_data_str: StrKeyDict,
    discrete_data_int: IntKeyDict,
    continuous_data: ContinuousType,
) -> None:
    for key, value in line.items():
        if isinstance(value, str):
            discrete_data_str.setdefault(value, []).append((nodeid, score))
        elif isinstance(value, int) and not isinstance(value, bool):
            discrete_data_int.setdefault(value, []).append((nodeid, score))
        elif isinstance(value, float):
            continuous_data.setdefault(key, {}).setdefault(value, []).append(
                (nodeid, score)
            )
        elif isinstance(value, bool):
            discrete_data_str.setdefault(str(value), []).append((nodeid, score))
        elif isinstance(value, list):
            if all(isinstance(v, (int, float)) for v in value):
                for v in value:
                    continuous_data.setdefault(key, {}).setdefault(float(v), []).append(
                        (nodeid, score)
                    )


# ========================================================================
#  Splitters
# ========================================================================


def split_continuous(
    data: ContinuousType,
) -> tuple[ContinuousType, ContinuousType]:
    bounded: ContinuousType = {}
    unbounded: ContinuousType = {}
    for key, vd in data.items():
        vals = list(vd.keys())
        if vals and all(0 <= v <= 1 for v in vals):
            bounded[key] = vd
        else:
            unbounded[key] = vd
    return bounded, unbounded


def split_positional(
    data: PositionalType,
) -> tuple[PositionalType, PositionalType]:
    bounded: PositionalType = {}
    unbounded: PositionalType = {}
    for key, inner in data.items():
        all_vals = []
        for field in ("x", "y", "w", "h"):
            if field in inner:
                all_vals.extend(inner[field].keys())
        if all_vals and all(0 <= v <= 1 for v in all_vals):
            bounded[key] = inner
        elif all_vals:
            unbounded[key] = inner
    return bounded, unbounded


def split_discrete(
    data: dict[str | int, ElementList],
) -> tuple[dict[str | int, ElementList], dict[str | int, ElementList]]:
    numeric: dict[str | int, ElementList] = {}
    labels: dict[str | int, ElementList] = {}
    for k, v in data.items():
        if isinstance(k, (int, float)):
            numeric[k] = v
        else:
            labels[k] = v
    return numeric, labels


# ========================================================================
#  Analysis helpers
# ========================================================================

MIN_BUCKET_COUNT = 10


def _fmt_continuous_bucket(value: float, prefix: str = "") -> str:
    label = f"value_{float(value):.2f}"
    return f"{prefix}{label}" if prefix else label


def _fmt_position_bucket(position: int) -> str:
    pos = int(position)
    if pos <= 10:
        start = 0
        end = 10
    else:
        start = ((pos - 1) // 10) * 10 + 1
        end = start + 9
    return f"position_{start}_to_{end}"


def _safe_stats(scores: list[float], values: list[float] | None = None) -> dict | None:
    s = np.array(scores, dtype=np.float64) if scores else np.array([])
    v = np.array(values, dtype=np.float64) if values else np.array([])
    if len(s) < MIN_BUCKET_COUNT:
        return None
    sq = np.percentile(s, [25, 50, 75]) if len(s) > 1 else [s[0], s[0], s[0]]
    std_s = float(np.std(s)) if len(s) > 1 else 0.0
    leaf: dict[str, Any] = {
        "count": int(len(s)),
        "mean_score": round(float(np.mean(s)), 6),
        "std_score": round(std_s, 6),
        "min_score": round(float(np.min(s)), 6),
        "max_score": round(float(np.max(s)), 6),
        "p25_score": round(float(sq[0]), 6),
        "p50_score": round(float(sq[1]), 6),
        "p75_score": round(float(sq[2]), 6),
    }
    if len(v) > 0:
        std_v = float(np.std(v)) if len(v) > 1 else 0.0
        leaf["mean_value"] = round(float(np.mean(v)), 6)
        leaf["std_value"] = round(std_v, 6)
        leaf["min_value"] = round(float(np.min(v)), 6)
        leaf["max_value"] = round(float(np.max(v)), 6)
    else:
        leaf["mean_value"] = leaf["std_value"] = None
        leaf["min_value"] = leaf["max_value"] = None
    return leaf


def _extract_scores(
    entries: list[tuple[int, float]],
) -> list[float]:
    return [s for _, s in entries]


def bucket_continuous(
    value_scores: dict[float, list[tuple[int, float]]], prefix: str = ""
) -> dict[str, dict]:
    if not value_scores:
        return {}
    bucketed: dict[str, dict[str, list[float]]] = {}
    for value, element_list in value_scores.items():
        if not element_list:
            continue
        bucket_key = _fmt_continuous_bucket(value, prefix)
        entry = bucketed.setdefault(bucket_key, {"scores": [], "values": []})
        scores = [s for _, s in element_list]
        entry["scores"].extend(scores)
        entry["values"].extend([float(value)] * len(scores))
    buckets = {}
    for bucket_key, entry in bucketed.items():
        scores = entry["scores"]
        values = entry["values"]
        if len(scores) < MIN_BUCKET_COUNT:
            continue
        leaf = _safe_stats(scores, values)
        if leaf:
            buckets[bucket_key] = leaf
    return buckets


def param_to_buckets(
    data: dict[str, dict[float, list[tuple[int, float]]]], prefix: str | None = None
) -> dict[str, dict]:
    return {
        name: bucket_continuous(
            value_scores,
            prefix=f"{name}_" if prefix is None else prefix,
        )
        for name, value_scores in data.items()
        if value_scores
    }


def discrete_to_buckets(
    data: dict[str | int, ElementList],
) -> dict[str, dict]:
    buckets = {}
    for val, scrs in data.items():
        if len(scrs) < MIN_BUCKET_COUNT:
            continue
        leaf = _safe_stats(_extract_scores(scrs))
        if leaf:
            buckets[f"value_{val}"] = leaf
    return buckets


def prompt_to_buckets(term_data: dict[str, object]) -> dict[str, dict]:
    result = {}
    for term, weight_map in term_data.items():
        if not isinstance(weight_map, dict):
            continue
        term_buckets = {}
        for weight, pos_map in weight_map.items():
            if not isinstance(pos_map, dict):
                continue
            pos_buckets = {}
            for pos, scores in pos_map.items():
                if not isinstance(scores, list):
                    continue
                flat_scores = [s for _, s in scores if isinstance(s, (int, float))]
                leaf = _safe_stats(flat_scores)
                if leaf:
                    pos_buckets[_fmt_position_bucket(int(pos))] = leaf
            if pos_buckets:
                term_buckets[f"weight_{weight}"] = pos_buckets
        if term_buckets:
            result[term] = term_buckets
    return result


def positional_to_buckets(
    pos_data: PositionalType,
) -> dict[str, dict]:
    result = {}
    for field, inner in pos_data.items():
        comps = {}
        for c in ("x", "y", "w", "h"):
            value_scores = inner.get(c, {})
            if not value_scores:
                continue
            comps[c] = bucket_continuous(value_scores)
        if comps:
            result[field] = comps
    return result


# Raw grouping helpers


def _group_continuous_raw(
    data: ContinuousType,
) -> dict[str, FloatKeyDict]:
    return {name: {round(val, 2): list(entries) for val, entries in vd.items()} for name, vd in data.items()}


def _group_str_discrete_raw(
    data: StrKeyDict,
) -> StrKeyDict:
    return {k: list(v) for k, v in data.items()}


def _group_int_discrete_raw(
    data: IntKeyDict,
) -> IntKeyDict:
    return {k: list(v) for k, v in data.items()}


def _group_positional_raw(
    data: PositionalType,
) -> dict[str, dict[str, FloatKeyDict]]:
    result = {}
    for field, inner in data.items():
        comps = {}
        for comp in ("x", "y", "w", "h"):
            buckets = inner.get(comp, {})
            if buckets:
                comps[comp] = {round(val, 2): list(entries) for val, entries in buckets.items()}
        if comps:
            result[field] = comps
    return result


def _group_prompt_term_raw(
    term_data: PromptType,
) -> dict[str, dict[float, dict[int, ElementList]]]:
    result = {}
    for term, weight_map in term_data.items():
        weight_entries = {}
        for weight, pos_map in weight_map.items():
            pos_entries = {pos: list(scores) for pos, scores in pos_map.items()}
            weight_entries[round(weight, 2)] = pos_entries
        if weight_entries:
            result[term] = weight_entries
    return result


# _split_continuous is now handled by split_continuous (no underscore version)


# Flatten helpers


def flatten_results(nested: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for k, v in nested.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict) and "count" in v and "mean_score" in v:
            result[key] = v
        elif isinstance(v, dict):
            result.update(flatten_results(v, key))
    return result


def flatten_to_jsonl_entries(flat: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"key": k, **v} for k, v in flat.items()]


# Main analysis pipeline


def extract_all_data(
    text_data: list[dict[str, Any]],
    scores: list[float],
) -> dict[str, Any]:
    lora_data: ContinuousType = {}
    positive_prompt: PromptType = {}
    negative_prompt: PromptType = {}
    continuous_data: ContinuousType = {}
    face_age_data: ContinuousType = {}
    face_gender_data: ContinuousType = {}
    face_race_data: ContinuousType = {}
    pose_data: ContinuousType = {}
    lh_data: ContinuousType = {}
    rh_data: ContinuousType = {}
    positional_data: PositionalType = {}
    discrete_data_str: StrKeyDict = {}
    discrete_data_int: IntKeyDict = {}

    from tqdm import tqdm

    for i in tqdm(range(len(text_data)), desc="Processing", unit="line", delay=3.0):
        outer = text_data[i]
        current_score = float(scores[i])
        current_line = dict(outer[next(iter(outer))])

        extract_lora(current_line, current_score, i, lora_data)
        extract_prompts(
            current_line, current_score, i, positive_prompt, negative_prompt
        )
        extract_face_logits(
            current_line,
            current_score,
            i,
            face_age_data,
            face_gender_data,
            face_race_data,
        )
        extract_face_bbox(
            current_line, current_score, i, continuous_data, positional_data
        )
        extract_pose_landmarks(
            current_line, current_score, i, pose_data, positional_data
        )
        extract_hand_landmarks(
            current_line, current_score, i, lh_data, rh_data, positional_data
        )
        extract_image_sizes(
            current_line, current_score, i, continuous_data, positional_data
        )
        extract_remaining_fields(
            current_line, current_score, i, discrete_data_str, discrete_data_int, continuous_data
        )

    return {
        "lora_data": lora_data,
        "positive_prompt": positive_prompt,
        "negative_prompt": negative_prompt,
        "discrete_data_str": discrete_data_str,
        "discrete_data_int": discrete_data_int,
        "continuous_data": continuous_data,
        "face_age_data": face_age_data,
        "face_gender_data": face_gender_data,
        "face_race_data": face_race_data,
        "pose_data": pose_data,
        "lh_data": lh_data,
        "rh_data": rh_data,
        "positional_data": positional_data,
    }


def run_full_analysis(
    text_data: list[dict[str, Any]],
    scores: list[float],
) -> dict[str, dict]:
    raw = extract_all_data(text_data, scores)

    positive_prompt = raw["positive_prompt"]
    negative_prompt = raw["negative_prompt"]

    bounded, unbounded = split_continuous(raw["continuous_data"])

    all_results: dict[str, dict] = {}
    all_c = dict(bounded)
    all_c.update(unbounded)
    all_results["continuous"] = param_to_buckets(all_c)
    all_results["lora"] = param_to_buckets(raw["lora_data"], prefix="weight_")
    all_results["discrete_numeric"] = discrete_to_buckets(raw["discrete_data_int"])
    all_results["discrete_labels"] = discrete_to_buckets(raw["discrete_data_str"])
    all_results["face_age"] = param_to_buckets(raw["face_age_data"])
    all_results["face_gender"] = param_to_buckets(raw["face_gender_data"])
    all_results["face_race"] = param_to_buckets(raw["face_race_data"])
    all_results["pose"] = param_to_buckets(raw["pose_data"])
    all_results["left_hand"] = param_to_buckets(raw["lh_data"])
    all_results["right_hand"] = param_to_buckets(raw["rh_data"])
    all_results["positional"] = positional_to_buckets(raw["positional_data"])
    all_results["prompt_terms"] = {
        "positive": prompt_to_buckets(positive_prompt),
        "negative": prompt_to_buckets(negative_prompt),
    }

    return all_results


def build_raw_values(
    raw: dict[str, Any],
) -> dict[str, Any]:
    return {
        "continuous": _group_continuous_raw(raw["continuous_data"]),
        "lora": _group_continuous_raw(raw["lora_data"]),
        "discrete": _group_str_discrete_raw(raw["discrete_data_str"]),
        "discrete_int": _group_int_discrete_raw(raw["discrete_data_int"]),
        "face_age": _group_continuous_raw(raw["face_age_data"]),
        "face_gender": _group_continuous_raw(raw["face_gender_data"]),
        "face_race": _group_continuous_raw(raw["face_race_data"]),
        "pose": _group_continuous_raw(raw["pose_data"]),
        "left_hand": _group_continuous_raw(raw["lh_data"]),
        "right_hand": _group_continuous_raw(raw["rh_data"]),
        "positional": _group_positional_raw(raw["positional_data"]),
        "prompt_terms": {
            "positive": _group_prompt_term_raw(raw["positive_prompt"]),
            "negative": _group_prompt_term_raw(raw["negative_prompt"]),
        },
    }
