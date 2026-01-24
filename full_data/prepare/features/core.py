from typing import List, Dict, Any, Tuple

from .meta import parse_entry_meta
from .terms import extract_terms, map_terms_to_indices, get_categorical_indices
from .assemble import assemble_feature_vector
from .embeddings import encode_prompt


def build_feature_vector(
    entry: Dict[str, Any],
    slots: Dict[str, int],
    mode: str,
    dim: int,
    normalization: Dict[str, Any],
    order: List[str],
    embedding_model: object = None,
    precomputed_embeddings: Tuple[Any, Any] = (None, None),
) -> Tuple[List[float], Dict[str, Any], List[str]]:
    (
        cfg_norm,
        steps_norm,
        lora_weight,
        sampler_name,
        scheduler_name,
        model_name,
        lora_name,
        width_norm,
        height_norm,
        ar_norm,
    ) = parse_entry_meta(entry, normalization)
    cat_indices, cat_statuses = get_categorical_indices(
        slots, sampler_name, scheduler_name, model_name, lora_name
    )
    sampler_idx, scheduler_idx, model_idx, lora_idx = cat_indices
    sampler_status, scheduler_status, model_status, lora_status = cat_statuses
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
    if mode == "embedding":
        pos_emb_pre, neg_emb_pre = precomputed_embeddings
        if pos_emb_pre is not None:
             pos_emb = pos_emb_pre
        else:
             pos_emb = encode_prompt(entry.get("positive_prompt", ""), embedding_model)
        
        if neg_emb_pre is not None:
             neg_emb = neg_emb_pre
        else:
             neg_emb = encode_prompt(entry.get("negative_prompt", ""), embedding_model)

        pos_indices, pos_statuses = (pos_emb, [])
        neg_indices, neg_statuses = (neg_emb, [])
    else:
        pos_indices, pos_statuses = map_terms_to_indices(
            pos_terms, "positive_terms", slots["positive_terms"]
        )
        neg_indices, neg_statuses = map_terms_to_indices(
            neg_terms, "negative_terms", slots["negative_terms"]
        )
    feature, overflowed = assemble_feature_vector(
        cfg_norm,
        steps_norm,
        lora_weight,
        sampler_idx,
        scheduler_idx,
        model_idx,
        lora_idx,
        width_norm,
        height_norm,
        ar_norm,
        pos_indices,
        neg_indices,
        slots,
        normalization,
        entry,
        sampler_status,
        scheduler_status,
        model_status,
        lora_status,
        pos_statuses,
        neg_statuses,
        order,
        mode,
        dim,
    )
    return feature, slots, overflowed
