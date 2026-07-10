import re
from dataclasses import dataclass, field
from typing import cast

WeightedTerm = tuple[str, float, int]


@dataclass
class ExtractionResult:
    terms: list[WeightedTerm]
    raw: list[WeightedTerm] = field(default_factory=lambda: cast(list[WeightedTerm], []))
    filtered_out: list[WeightedTerm] = field(default_factory=lambda: cast(list[WeightedTerm], []))
    stripped: list[WeightedTerm] = field(default_factory=lambda: cast(list[WeightedTerm], []))
    duplicates: list[WeightedTerm] = field(default_factory=lambda: cast(list[WeightedTerm], []))


def extract_weight_from_paren(text: str) -> tuple[str, float]:
    """
    Extracts content and weight from (term:weight) or (term).
    Implicit parentheses multiply the weight by 1.1.
    """
    s: str = text.strip()

    # Matches (content : 1.5) or (content:1.5)
    match: re.Match[str] | None = re.match(r"^\(\s*(.*?)\s*:\s*(\d+\.?\d*)\s*\)$", s)
    if match:
        return (match.group(1).strip(), float(match.group(2)))

    # Matches simple (content)
    if s.startswith("(") and s.endswith(")"):
        return (s[1:-1].strip(), 1.1)

    return (s, 1.0)


def tokenize_by_depth(text: str, splitters: set[str]) -> list[str]:
    """
    Splits text by splitters and parenthetical boundaries, respecting nesting depth.
    Ensures phrases stay together unless a splitter is encountered at the top level.
    """
    tokens: list[str] = []
    current_chunk: list[str] = []
    depth = 0
    i = 0
    length: int = len(text)

    # Sort splitters to check longer word-based ones (like 'but') before characters
    word_splitters: list[str] = sorted(
        [s for s in splitters if s.isalnum()], key=len, reverse=True
    )

    while i < length:
        char: str = text[i]

        if char == "(":
            if depth == 0 and current_chunk:
                tokens.append("".join(current_chunk).strip())
                current_chunk = []
            depth += 1
            current_chunk.append(char)
        elif char == ")":
            if depth > 0:
                depth -= 1
            current_chunk.append(char)
            if depth == 0:
                tokens.append("".join(current_chunk).strip())
                current_chunk = []
        elif depth == 0:
            found_splitter = False

            # Check for non-alphanumeric splitters (like ',')
            if char in splitters and not char.isalnum():
                tokens.append("".join(current_chunk).strip())
                current_chunk = []
                found_splitter = True

            # Check for word-based splitters (like 'but') with boundary check
            elif char.isspace() or i == 0:
                lookahead: str = text[i:].strip()
                for ws in word_splitters:
                    if lookahead.startswith(ws):
                        # Verify it's a whole word, not part of another word
                        end_idx: int = i + text[i:].find(ws) + len(ws)
                        if end_idx == length or not text[end_idx].isalnum():
                            tokens.append("".join(current_chunk).strip())
                            current_chunk = []
                            # Skip the length of the splitter word
                            i += text[i:].find(ws) + len(ws) - 1
                            found_splitter = True
                            break

            if not found_splitter:
                current_chunk.append(char)
        else:
            current_chunk.append(char)
        i += 1

    if current_chunk:
        tokens.append("".join(current_chunk).strip())

    return [t for t in tokens if t]


def clean_term(term: str) -> str:
    """Normalizes string content and removes technical artifacts."""
    s: str = term.strip()
    s: str = s.replace("\\", "")
    s: str = re.sub(r"\s+", " ", s)
    s: str = s.replace("|", "")
    # Remove surrounding punctuation/noise common in prompt tags
    s: str = s.strip(" \t\n\r\"'`,;:.!?#<>[]{}()")
    # Clean trailing weight markers that might be left over
    s: str = re.sub(r":\s*\d+\.?\d*$", "", s)
    return s.lower()


def filter_terms(
    terms: list[WeightedTerm],
    connectors: set[str],
    splitters: set[str],
) -> tuple[list[WeightedTerm], list[WeightedTerm]]:
    """Removes standard stopwords unless they are protected by the user's sets.
    Returns (kept, filtered_out)."""
    stopwords: set[str] = {"a", "an", "the", "in", "is", "at", "to", "by", "of"}

    kept: list[WeightedTerm] = []
    filtered: list[WeightedTerm] = []
    for term, weight, idx in terms:
        if " " not in term:
            if term in stopwords and term not in connectors and term not in splitters:
                filtered.append((term, weight, idx))
                continue
        if term:
            kept.append((term, weight, idx))
    return kept, filtered


def deduplicate_terms(
    terms: list[WeightedTerm],
) -> tuple[list[WeightedTerm], list[WeightedTerm]]:
    """Merges duplicate terms, retaining the highest weight found.
    Returns (deduplicated, duplicates_removed)."""
    best: dict[str, tuple[float, int]] = {}
    duplicates: list[WeightedTerm] = []
    for term, weight, idx in terms:
        if term in best:
            existing_weight, existing_idx = best[term]
            if weight > existing_weight:
                duplicates.append((term, existing_weight, existing_idx))
                best[term] = (weight, idx)
            else:
                duplicates.append((term, weight, idx))
        else:
            best[term] = (weight, idx)
    result: list[WeightedTerm] = [(t, w, i) for t, (w, i) in best.items()]
    return result, duplicates


def _extract_recursive(
    text: str, current_weight: float, splitters: set[str]
) -> list[tuple[str, float]]:
    """Handles the heavy lifting of nesting and weight multiplication."""
    text = text.strip()
    if not text:
        return []

    if text.startswith("(") and text.endswith(")"):
        content, weight_val = extract_weight_from_paren(text)

        if ":" in text:
            new_weight: float = weight_val
        else:
            new_weight: float = current_weight * weight_val

        return _extract_recursive(content, new_weight, splitters)

    chunks: list[str] = tokenize_by_depth(text, splitters)
    results: list[tuple[str, float]] = []
    for chunk in chunks:
        if chunk.startswith("(") and chunk.endswith(")"):
            results.extend(_extract_recursive(chunk, current_weight, splitters))
        else:
            results.append((chunk, current_weight))
    return results


def extract_terms(
    text: str,
    connectors: tuple[str, ...] = ("and", "or"),
    splitters: tuple[str, ...] = (",", "but", "not"),
) -> ExtractionResult:
    """
    The main entry point for processing prompt text into weighted vectors.
    """
    if not text:
        return ExtractionResult(terms=[])

    conn_set: set[str] = set(connectors)
    split_set: set[str] = set(splitters)

    # 1. Recursive parsing of parentheses and weights
    raw = _extract_recursive(text, 1.0, split_set)
    raw_indexed: list[WeightedTerm] = [(t, w, i) for i, (t, w) in enumerate(raw)]

    # 2. Clean and normalize the strings, capturing any that become empty
    stripped: list[WeightedTerm] = []
    cleaned_list: list[WeightedTerm] = []
    for t, w, i in raw_indexed:
        ct = clean_term(t)
        entry = (ct, w, i)
        if ct:
            cleaned_list.append(entry)
        else:
            stripped.append(entry)

    # 3. Filter out noise words
    kept, filtered_out = filter_terms(cleaned_list, conn_set, split_set)

    # 4. Final deduplication
    deduped, duplicates = deduplicate_terms(kept)

    return ExtractionResult(
        terms=deduped,
        raw=raw_indexed,
        filtered_out=filtered_out,
        stripped=stripped,
        duplicates=duplicates,
    )
