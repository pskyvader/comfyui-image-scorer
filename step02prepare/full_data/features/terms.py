from typing import List, Tuple, Set, Dict
import re
from step02prepare.full_data.config.maps import get_or_add

WeightedTerm = Tuple[str, float]


def extract_weight_from_paren(text: str) -> Tuple[str, float]:
    match = re.match(r"^\((.+?):(\d+\.?\d*)\)$", text.strip())
    if match:
        return (match.group(1).strip(), float(match.group(2)))
    stripped = text.strip()
    if stripped.startswith("(") and stripped.endswith(")"):
        return (stripped[1:-1].strip(), 1.0)
    return (stripped, 1.0)


def split_weighted_compound(text: str, weight: float) -> List[WeightedTerm]:
    text = text.strip()
    if not text:
        return []
    parts = re.split(r"\s*,\s*", text)
    results: List[WeightedTerm] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part.startswith("(") and ":" in part:
            inner_term, inner_weight = extract_weight_from_paren(part)
            if inner_term != part:
                results.extend(split_weighted_compound(inner_term, inner_weight))
            else:
                # If term didn't change (e.g. mismatched parens), treat as plain text
                words = part.split()
                for word in words:
                    word = word.strip()
                    if word and len(word) > 1:
                        results.append((word.lower(), weight))
        else:
            words = part.split()
            for word in words:
                word = word.strip()
                if word and len(word) > 1:
                    results.append((word.lower(), weight))
    return results


def parse_parenthesized_term(text: str) -> List[WeightedTerm]:
    term, weight = extract_weight_from_paren(text)
    return split_weighted_compound(term, weight)


def tokenize_text(text: str, splitters: Set[str]) -> List[str]:
    # Preserve parenthesized groups as tokens
    tokens: List[str] = []
    buf: List[str] = []
    i = 0
    L = len(text)

    def flush_buf() -> None:
        s = "".join(buf).strip()
        if s:
            tokens.append(s)
        buf.clear()

    while i < L:
        ch = text[i]
        if ch == "(":
            flush_buf()
            start = i
            depth = 0
            while i < L:
                if text[i] == "(":
                    depth += 1
                elif text[i] == ")":
                    depth -= 1
                    if depth == 0:
                        i += 1
                        break
                i += 1
            paren = text[start:i].strip()
            tokens.append(paren)
            continue
        if ch in ";[]":
            flush_buf()
            i += 1
            continue
        if ch == ",":
            if "," in splitters:
                flush_buf()
            i += 1
            continue
        buf.append(ch)
        i += 1
    flush_buf()
    return tokens


def clean_term(term: str) -> str:
    s = term.strip()
    s = s.replace("\\", "")
    s = re.sub(r"\s+", " ", s)
    s = s.replace("|", "")
    s = s.strip(" \t\n\r\"'`,;:.!?#<>[]{}")
    s = re.sub(r":\d+\.?\d*$", "", s)
    s = s.lower()
    return s


def filter_terms(
    terms: List[WeightedTerm], connectors: Set[str], splitters: Set[str]
) -> List[WeightedTerm]:
    always_remove = {"a", "an", "the", "in", "is", "at", "to", "by", "of", ""}
    filtered: List[WeightedTerm] = []
    for term, weight in terms:
        term = clean_term(term)
        if len(term) <= 1:
            continue
        if term in always_remove and term not in connectors and term not in splitters:
            continue
        if term:
            filtered.append((term, weight))
    return filtered


def deduplicate_terms(terms: List[WeightedTerm]) -> List[WeightedTerm]:
    term_weights: Dict[str, float] = {}
    for term, weight in terms:
        if term in term_weights:
            term_weights[term] = max(term_weights[term], weight)
        else:
            term_weights[term] = weight
    return list(term_weights.items())


def extract_terms(
    text: str, connectors: Set[str] | None = None, splitters: Set[str] | None = None
) -> List[WeightedTerm]:
    if not text:
        return []
    if connectors is None:
        connectors = {"and", "or"}
    if splitters is None:
        splitters = {"but", "not", ",", "girl", "years"}
    tokens = tokenize_text(text, splitters)
    all_terms: List[WeightedTerm] = []
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        if token.startswith("(") and ")" in token:
            weighted_terms = parse_parenthesized_term(token)
            all_terms.extend(weighted_terms)
        else:
            words = re.split(r"[\s,]+", token)
            for word in words:
                word = word.strip()
                if word:
                    all_terms.append((word.lower(), 1.0))
    filtered = filter_terms(all_terms, connectors, splitters)
    deduped = deduplicate_terms(filtered)
    return deduped


def map_terms_to_indices(
    terms: List[WeightedTerm], map_name: str, max_slots: int
) -> Tuple[List[Tuple[int, float]], List[str]]:
    indexed: List[Tuple[int, float]] = []
    statuses: List[str] = []
    for term, weight in terms:
        idx, _, st = get_or_add(map_name, term, max_slots)
        indexed.append((idx, weight))
        statuses.append(st)
    return (indexed, statuses)


def get_categorical_indices(
    slots: Dict[str, int],
    sampler_name: str | None,
    scheduler_name: str | None,
    model_name: str | None,
    lora_name: str | None,
) -> Tuple[Tuple[int, int, int, int], Tuple[str, str, str, str]]:
    sampler_idx, _, sampler_st = get_or_add(
        "sampler", sampler_name or "", slots["sampler"] if "sampler" in slots else 0
    )
    scheduler_idx, _, scheduler_st = get_or_add(
        "scheduler", scheduler_name or "", slots["scheduler"] if "scheduler" in slots else 0
    )
    model_idx, _, model_st = get_or_add(
        "model", model_name or "", slots["model"] if "model" in slots else 0
    )
    lora_idx, _, lora_st = get_or_add("lora", lora_name or "", slots["lora"] if "lora" in slots else 0)
    return (
        (sampler_idx, scheduler_idx, model_idx, lora_idx),
        (sampler_st, scheduler_st, model_st, lora_st),
    )
