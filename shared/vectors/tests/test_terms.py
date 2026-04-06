import sys
from pathlib import Path
import pytest

# Add the parent directory to sys.path so we can import the module
# assuming the main file is named `terms.py`
sys.path.append(str(Path(__file__).parent.parent))

from terms import (
    extract_weight_from_paren,
    clean_term,
    filter_terms,
    deduplicate_terms,
    extract_terms,
)

# =================================================================
# ORIGINAL INTEGRATION TESTS
# =================================================================


long_text_test_list: list[tuple[str, list[tuple[str, float]]]] = []

long_text_test_list.append(
    ("(a fast car:1.2)", [("a fast car", 1.2)]),
)

long_text_test_list.append(
    ("(red, blue:1.5)", [("red", 1.5), ("blue", 1.5)]),
)


long_text_test_list.append(
    (
        "(a very long phrase with multiple words:0.8)",
        [("a very long phrase with multiple words", 0.8)],
    ),
)

long_text_test_list.append(
    (
        "(  a very long phrase with multiple words  :  0.8  )",
        [("a very long phrase with multiple words", 0.8)],
    ),
)

long_text_test_list.append(
    (
        "a phrase with commas, and (different weights:1.0) in same (sentence)",
        [
            ("a phrase with commas", 1.0),
            ("and", 1.0),
            ("different weights", 1.0),
            ("in same", 1.0),
            ("sentence", 1.1),
        ],
    ),
)

long_text_test_list.append(
    (
        "a very long phrase with multiple words and no parentheses",
        [("a very long phrase with multiple words and no parentheses", 1.0)],
    )
)

long_text_test_list.append(
    (
        "a phrase with connectors and separators but no parenthesis",
        [
            ("a phrase with connectors and separators", 1.0),
            ("no parenthesis", 1.0),
        ],
    )
)

long_text_test_list.append(
    (
        "connectors, separators and no pharentesis but  example and spaces",
        [
            ("connectors", 1.0),
            ("separators and no pharentesis", 1.0),
            ("example and spaces", 1.0),
        ],
    )
)

long_text_test_list.append(
    (
        "(nested (parentheses (with many) weights))",
        [
            ("nested", 1.1),
            ("parentheses", 1.1**2),
            ("with many", 1.1**3),
            ("weights", 1.1**2),
        ],
    )
)

long_text_test_list.append(
    (
        "(nested connectors and conn (and cont) and connector but sep if not)",
        [
            ("nested connectors and conn", 1.1),
            ("and cont", 1.1**2),
            ("and connector", 1.1),
            ("sep if", 1.1),
        ],
    )
)


@pytest.mark.parametrize("input_text, expected_output", long_text_test_list)
def test_extract_terms_variations(
    input_text: str, expected_output: list[tuple[str, float]]
):
    assert extract_terms(input_text) == expected_output


def test_custom_splitters():
    # removed comma from splitters, now it should be treated like any regular character
    text = "term1, term2 but term3 not term4"
    expected = [("term1, term2", 1.0), ("term3", 1.0), ("term4", 1.0)]
    assert extract_terms(text, splitters=("but", "not")) == expected


def test_custom_connectors():
    # added "but" to connectors, now it should be kept as a term
    text = "term1 and term2 but term3 not term4"
    expected = [("term1 and term2 but term3", 1.0), ("term4", 1.0)]
    assert (
        extract_terms(text, connectors=("and", "but"), splitters=("not",)) == expected
    )


# =================================================================
# NEW UNIT TESTS FOR INDIVIDUAL FUNCTIONS
# =================================================================


# --- clean_term ---
@pytest.mark.parametrize(
    "input_term, expected",
    [
        ("  Term  ", "term"),
        ("Clean\\ing", "cleaning"),
        ("Weight:1.5", "weight"),
        ("Extra   Spaces", "extra spaces"),
        ("[brackets]!", "brackets"),
        ("|pipe|", "pipe"),
    ],
)
def test_clean_term(input_term: str, expected: str):
    """Verifies that strings are normalized correctly."""
    assert clean_term(input_term) == expected


# --- deduplicate_terms ---
def test_deduplicate_terms_logic():
    """Verifies that the highest weight is kept for duplicate terms."""
    input_data = [("apple", 1.0), ("orange", 1.2), ("apple", 1.5)]
    result = dict(deduplicate_terms(input_data))
    assert result["apple"] == 1.5
    assert result["orange"] == 1.2
    assert len(result) == 2


# --- filter_terms ---
def test_filter_terms_with_connectors():
    """Verifies stop-words are removed unless they are in connectors/splitters."""
    connectors = {"and"}
    splitters = {","}
    input_terms = [("the", 1.0), ("and", 1.0), ("car", 1.0), ("a", 1.0)]

    filtered = [t[0] for t in filter_terms(input_terms, connectors, splitters)]
    assert "and" in filtered
    assert "car" in filtered
    assert "the" not in filtered
    assert "a" not in filtered


# --- extract_weight_from_paren ---
@pytest.mark.parametrize(
    "text, expected",
    [
        ("(simple:1.5)", ("simple", 1.5)),
        ("(no_colon)", ("no_colon", 1.1)),
        ("(phrase with spaces:0.8)", ("phrase with spaces", 0.8)),
    ],
)
def test_extract_weight_from_paren(text: str, expected: tuple[str, float]):
    """Verifies weight extraction from parenthetical strings."""
    assert extract_weight_from_paren(text) == expected
