import re

WeightedTerm = tuple[str, float]


def extract_weight_from_paren(text: str) -> tuple[str, float]:
    match = re.match(r"^\((.+?):(\d+\.?\d*)\)$", text.strip())
    if match:
        return (match.group(1).strip(), float(match.group(2)))
    stripped = text.strip()
    if stripped.startswith("(") and stripped.endswith(")"):
        return (stripped[1:-1].strip(), 1.0)
    return (stripped, 1.0)


def split_weighted_compound(text: str, weight: float) -> list[WeightedTerm]:
    text = text.strip()
    if not text:
        return []
    parts = re.split(r"\s*,\s*", text)
    results: list[WeightedTerm] = []
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


def parse_parenthesized_term(text: str) -> list[WeightedTerm]:
    term, weight = extract_weight_from_paren(text)
    return split_weighted_compound(term, weight)


def tokenize_text(text: str, splitters: set[str]) -> list[str]:
    # Preserve parenthesized groups as tokens
    tokens: list[str] = []
    buf: list[str] = []
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
    terms: list[WeightedTerm], connectors: set[str], splitters: set[str]
) -> list[WeightedTerm]:
    always_remove = {"a", "an", "the", "in", "is", "at", "to", "by", "of", ""}
    filtered: list[WeightedTerm] = []
    for term, weight in terms:
        term = clean_term(term)
        if len(term) <= 1:
            continue
        if term in always_remove and term not in connectors and term not in splitters:
            continue
        if term:
            filtered.append((term, weight))
    return filtered


def deduplicate_terms(terms: list[WeightedTerm]) -> list[WeightedTerm]:
    term_weights: dict[str, float] = {}
    for term, weight in terms:
        if term in term_weights:
            term_weights[term] = max(term_weights[term], weight)
        else:
            term_weights[term] = weight
    return list(term_weights.items())


def extract_terms(
    text: str, connectors: set[str] | None = None, splitters: set[str] | None = None
) -> list[WeightedTerm]:
    if not text:
        return []
    if connectors is None:
        connectors = {"and", "or"}
    if splitters is None:
        splitters = {"but", "not", ",", "and", "or"}
    tokens = tokenize_text(text, splitters)
    all_terms: list[WeightedTerm] = []
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
