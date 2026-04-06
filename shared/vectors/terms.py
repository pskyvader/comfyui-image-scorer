import re

# Type alias for clarity
WeightedTerm = tuple[str, float]


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
    s: str = s.strip(" \t\n\r\"'`,;:.!?#<>[]{}")
    # Clean trailing weight markers that might be left over
    s: str = re.sub(r":\s*\d+\.?\d*$", "", s)
    return s.lower()


def filter_terms(
    terms: list[WeightedTerm], connectors: set[str], splitters: set[str]
) -> list[WeightedTerm]:
    """Removes standard stopwords unless they are protected by the user's sets."""
    stopwords: set[str] = {"a", "an", "the", "in", "is", "at", "to", "by", "of"}

    filtered: list[tuple[str, float]] = []
    for term, weight in terms:
        if " " not in term:
            if term in stopwords and term not in connectors and term not in splitters:
                continue
        if term:
            filtered.append((term, weight))
    return filtered


def deduplicate_terms(terms: list[WeightedTerm]) -> list[WeightedTerm]:
    """Merges duplicate terms, retaining the highest weight found."""
    max_weights: dict[str, float] = {}
    for term, weight in terms:
        max_weights[term] = max(max_weights.get(term, 0.0), weight)
    return list(max_weights.items())


def _extract_recursive(
    text: str, current_weight: float, splitters: set[str]
) -> list[WeightedTerm]:
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
    results: list[WeightedTerm] = []
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
) -> list[WeightedTerm]:
    """
    The main entry point for processing prompt text into weighted vectors.
    """
    if not text:
        return []

    # Convert tuples to sets for efficient O(1) lookups
    conn_set: set[str] = set(connectors)
    split_set: set[str] = set(splitters)

    # 1. Recursive parsing of parentheses and weights
    raw_weighted_list: list[tuple[str, float]] = _extract_recursive(
        text, 1.0, split_set
    )

    # 2. Clean and normalize the strings
    cleaned_list: list[tuple[str, float]] = [
        (clean_term(t), w) for t, w in raw_weighted_list
    ]

    # 3. Filter out noise words
    filtered_list: list[tuple[str, float]] = filter_terms(
        cleaned_list, conn_set, split_set
    )

    # 4. Final deduplication
    return deduplicate_terms(filtered_list)
