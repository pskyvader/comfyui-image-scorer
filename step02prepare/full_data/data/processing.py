import os
from typing import Any, Dict, List, Set, Tuple, cast
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path
import shutil

print("Importing shared modules...")
from shared.config import config
from shared.io import load_json_list_robust
from shared.paths import (
    prepare_output_dir,
    maps_dir,
    vectors_file,
    scores_file,
    index_file,
    training_model,
    processed_data,
    filtered_data,
    interaction_data,
)
print("Importing config modules...")
from step02prepare.full_data.config.manager import save_vector_schema
from step02prepare.full_data.config.schema import (
    get_vector_order,
    get_total_vector_length,
)
print("Importing data processing modules...")
from step02prepare.full_data.data.manager import (
    load_index,
    encode_images_to_vectors,
    pad_existing_vectors,
)
from step02prepare.full_data.data.metadata import load_metadata_entry

print("Importing feature modules...")
from step02prepare.full_data.features.core import build_feature_vector
from step02prepare.full_data.features.embeddings import load_model


def clean_training_artifacts() -> None:
    for p in [
        training_model,
        processed_data,
        filtered_data,
        interaction_data,
    ]:
        p = Path(p)
        if os.path.exists(p):
            print(f"Removing training artifact: {p}")
            try:
                os.remove(p)
            except OSError as e:
                print(f"Warning: Could not remove {p}: {e}")


def remove_existing_outputs() -> None:
    for p in [prepare_output_dir, maps_dir]:
        ppath = Path(p)
        if ppath.exists():
            print(f"Removing {p}")
            try:
                shutil.rmtree(ppath)
            except OSError as e:
                print(f"Warning: Could not remove {p}: {e}")
    clean_training_artifacts()


def load_existing_data() -> Tuple[List[str], List[List[float]], List[float]]:
    index_list = load_index(index_file)
    vectors_data = load_json_list_robust(vectors_file)
    vectors_list: List[List[float]] = []
    for row in vectors_data:
        if isinstance(row, list):
            vectors_list.append([float(v) for v in row])
    scores_data = load_json_list_robust(scores_file)
    scores_list: List[float] = [float(s) for s in scores_data]
    return index_list, vectors_list, scores_list


def collect_valid_files(
    files: List[Tuple[str, str]],
    processed_files: Set[str],
    error_log: List[Dict[str, str]],
) -> List[Tuple[str, Dict[str, Any], str, str]]:
    collected_data: List[Tuple[str, Dict[str, Any], str, str]] = []
    IMAGE_ROOT = config["image_root"]

    for img_path, meta_path in files:
        file_id = os.path.relpath(img_path, IMAGE_ROOT).replace("\\", "/")
        if file_id in processed_files:
            continue
        entry, ts, err = load_metadata_entry(meta_path)
        if entry is None or "score" not in entry:
            if err:
                error_log.append({"file": file_id, "reason": err})
            continue

        # Read image dimensions
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                entry["width"] = w
                entry["height"] = h
                entry["aspect_ratio"] = round(w / h, 4) if h > 0 else 0.0
        except Exception as e:
            error_log.append({"file": file_id, "reason": f"image_read_error: {e}"})
            continue

        ts_str = cast(str, ts)
        collected_data.append((img_path, entry, ts_str, file_id))
    return collected_data


def encode_new_images(
    collected_data: List[Tuple[str, Dict[str, Any], str, str]],
) -> List[List[float]]:
    if not collected_data:
        return []
    img_paths = [img_path for img_path, _, _, _ in collected_data]
    batch_size = int(config["vision_model"]["encoding_batch_size"])
    print(f"Encoding {len(img_paths)} images in batches of {batch_size}...")

    image_vectors = encode_images_to_vectors(img_paths, batch_size=batch_size)
    print("Encoding complete.")
    return image_vectors


def process_and_append_data(
    collected_data: List[Tuple[str, Dict[str, Any], str, str]],
    image_vectors: List[List[float]],
    vectors_list: List[List[float]],
    scores_list: List[float],
    index_list: List[str],
    processed_files: Set[str],
    schema: Dict[str, Any],
) -> None:
    print("Processing and appending new data...")
    total = len(collected_data)
    VECTORS_FILE = config["vectors_file"]
    prompt_representation = config["prompt_representation"]
    normalization = config["normalization"]
    cfg_slots = schema["slots"]

    order = get_vector_order()

    embedding_model = None
    if prompt_representation["mode"] == "embedding":
        embedding_model = load_model(
            prompt_representation["model"],
            int(prompt_representation["dim"]),
            config["vision_model"]["device"],
        )

    batch_size = int(config["vision_model"]["encoding_batch_size"])

    with tqdm(total=total, desc="Processing", unit="item") as pbar:
        for i in range(0, total, batch_size):
            batch = collected_data[i : i + batch_size]
            batch_indices = range(i, i + len(batch))

            batch_pos_embs = [None] * len(batch)
            batch_neg_embs = [None] * len(batch)

            if embedding_model is not None:
                pos_prompts = [item[1].get("positive_prompt", "") for item in batch]
                neg_prompts = [item[1].get("negative_prompt", "") for item in batch]

                # Batch encode
                # SentenceTransformer.encode returns ndarray if valid input logic
                pos_enc = embedding_model.encode(pos_prompts)
                neg_enc = embedding_model.encode(neg_prompts)

                # Cast to float32 to match encode_prompt behavior
                batch_pos_embs = [np.asarray(e, dtype=np.float32) for e in pos_enc]
                batch_neg_embs = [np.asarray(e, dtype=np.float32) for e in neg_enc]

            for j, (_, entry, ts, file_id) in enumerate(batch):
                idx = batch_indices[j]

                feature_vector, _, overflowed = build_feature_vector(
                    entry,
                    cfg_slots,
                    prompt_representation["mode"],
                    prompt_representation["dim"],
                    normalization,
                    order,
                    embedding_model,
                    precomputed_embeddings=(batch_pos_embs[j], batch_neg_embs[j]),
                )
                if overflowed:
                    changed = False
                    for name in overflowed:
                        prev = cfg_slots[name] if name in cfg_slots else 0
                        cfg_slots[name] = prev + 1
                        changed = True
                    if changed:
                        schema["slots"] = cfg_slots
                        save_vector_schema(schema)
                        pad_existing_vectors(
                            VECTORS_FILE, schema, prompt_representation
                        )
                        expected_len = get_total_vector_length(
                            cfg_slots,
                            prompt_representation["mode"],
                            prompt_representation["dim"],
                        )
                        for vec_idx, vec_existing in enumerate(vectors_list):
                            if len(vec_existing) < expected_len:
                                padding = [0.0] * (expected_len - len(vec_existing))
                                vectors_list[vec_idx] = vec_existing + padding
                    # Ensure the feature vector for the current item matches the updated schema
                    image_vector = image_vectors[idx]
                    feature_target_len = expected_len - len(image_vector)
                    if len(feature_vector) < feature_target_len:
                        feature_vector = feature_vector + [0.0] * (
                            feature_target_len - len(feature_vector)
                        )
                else:
                    image_vector = image_vectors[idx]

                full_vector = image_vector + feature_vector
                unique_id = f"{file_id}#{ts}"
                index_list.append(unique_id)
                vectors_list.append(full_vector)
                raw_score = entry.get("score")
                score_val = 0.0
                if raw_score is not None:
                    score_val = float(raw_score)
                scores_list.append(score_val)
                processed_files.add(file_id)
                pbar.update(1)


def check_for_leakage(
    vectors_list: List[List[float]], scores_list: List[float]
) -> None:
    if not vectors_list or not scores_list or len(vectors_list) != len(scores_list):
        return
    arr = np.array(vectors_list)
    y_arr = np.array(scores_list)
    corrs: List[float] = []
    for i in range(arr.shape[1]):
        col = arr[:, i]
        if np.allclose(col, col[0]):
            corrs.append(0.0)
            continue
        c = np.corrcoef(col, y_arr)[0, 1]
        corrs.append(float(c) if not np.isnan(c) else 0.0)
    corrs_arr = np.array(corrs)
    leak_cols = np.where(np.abs(corrs_arr) >= 0.9999)[0].tolist()
    eq_cols = [i for i in range(arr.shape[1]) if np.allclose(arr[:, i], y_arr)]
    leak_cols = sorted(set(leak_cols + eq_cols))
    if leak_cols:
        msg = (
            f"Detected feature columns that strongly match the target (possible leakage): {leak_cols}. "
            "Fix the feature assembly to avoid including target values as features."
        )
        print(msg)
        raise RuntimeError(msg)
