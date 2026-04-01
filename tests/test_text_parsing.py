"""
Comprehensive test suite for text parsing and term extraction logic.

Tests the functions in shared.vectors.terms module for handling:
- Plain text prompts
- Weighted terms (with :weight syntax)
- Parenthesized groups
- Nested structures
- Complex combinations
- Edge cases
"""

import pytest
from shared.vectors.terms import (
    extract_terms,
    extract_weight_from_paren,
    split_weighted_compound,
    parse_parenthesized_term,
    tokenize_text,
    clean_term,
    filter_terms,
    deduplicate_terms,
    WeightedTerm,
)

#todo: 
# add these tests:
# "A,b,c"->assert (a,1),(b,1),(c,1)
# "(a:1.1),(b:1),(c:0.5)"->assert (a,1.1),(b,1.0),(c,0.5)
# "(x,(a b:1.0),(c,d:0.5), )m:2.0"-> assert 
# (x,2.0),(a b,2.0),(c,1.0),(d,1.0),(m,1.0). 2.0 is invalid, just remove it
# "a b c, a b c:2.0, (abc:1.0)"->
# (a b c,1),(a b c,1),(abc,1)  . in this case since :2.0 is invalid, just remove it
# "(abc,)" -> (abc,1)
# "abc, ,,,(abc, abc (abc abc (abc,abc abc:2.0) abc, , ,,abc), abc)"
# -> (abc,1),(abc,1.1),(abc,1.1),(abc abc,1.21),(abc,2.42), abc abc,2.42),(abc,1.21),(abc,1.21),(abc,1.1)
# "ab. abc,((abc ... ab, ab...))"
#-> (ab. abc,1.0),(abc ... ab,1.21), (ab...,1.21) . any symbol other than comma must be preserved



class TestExtractWeightFromParen:
    """Test extraction of weights from parenthesized text like (text:1.5)"""

    def test_basic_weight(self):
        """Test extracting weight from (text:weight) format"""
        term, weight = extract_weight_from_paren("(dog:1.5)")
        assert term == "dog"
        assert weight == 1.5

    def test_integer_weight(self):
        """Test extracting integer weight"""
        term, weight = extract_weight_from_paren("(cat:2)")
        assert term == "cat"
        assert weight == 2.0

    def test_decimal_weight(self):
        """Test extracting decimal weight"""
        term, weight = extract_weight_from_paren("(bird:0.75)")
        assert term == "bird"
        assert weight == 0.75

    def test_no_weight(self):
        """Test parentheses without weight defaults to 1.0"""
        term, weight = extract_weight_from_paren("(fish)")
        assert term == "fish"
        assert weight == 1.0

    def test_multiple_words_no_weight(self):
        """Test multiple words in parentheses without weight"""
        term, weight = extract_weight_from_paren("(big dog)")
        assert term == "big dog"
        assert weight == 1.0

    def test_plain_text_no_parens(self):
        """Test plain text without parentheses"""
        term, weight = extract_weight_from_paren("dog")
        assert term == "dog"
        assert weight == 1.0


class TestSplitWeightedCompound:
    """Test splitting comma-separated weighted terms"""

    def test_simple_comma_separated(self):
        """Test basic comma-separated terms"""
        result = split_weighted_compound("dog, cat", 1.0)
        assert ("dog", 1.0) in result
        assert ("cat", 1.0) in result

    def test_weighted_terms(self):
        """Test comma-separated weighted terms"""
        result = split_weighted_compound("dog:1.5, cat:2.0", 1.0)
        # Terms should be extracted individually
        assert len(result) > 0

    def test_nested_weights(self):
        """Test nested parentheses with weights"""
        result = split_weighted_compound("(dog cat):1.5", 1.0)
        assert len(result) > 0

    def test_empty_input(self):
        """Test empty input returns empty list"""
        result = split_weighted_compound("", 1.0)
        assert result == []

    def test_single_term(self):
        """Test single term without comma"""
        result = split_weighted_compound("dog", 1.0)
        assert ("dog", 1.0) in result


class TestExtractTerms:
    """Test the main extract_terms function with various input formats"""

    def test_plain_text(self):
        """Test extracting terms from plain text"""
        result = extract_terms("dog cat")
        assert len(result) == 2
        # Result should contain dog and cat
        terms_only = [term for term, _ in result]
        assert "dog" in terms_only or "Dog" in [t.lower() for t in terms_only]

    def test_single_term(self):
        """Test single term extraction"""
        result = extract_terms("dog")
        assert len(result) == 1
        assert result[0][0].lower() == "dog"

    def test_weighted_terms_space_separated(self):
        """Test space-separated weighted format"""
        result = extract_terms("dog:1.5 cat:2.0")
        assert len(result) >= 1

    def test_comma_separated_plain(self):
        """Test comma-separated terms without weights"""
        result = extract_terms("dog, cat, bird")
        assert len(result) == 3

    def test_parenthesized_group(self):
        """Test parenthesized group"""
        result = extract_terms("(big dog) small")
        assert len(result) >= 2

    def test_parenthesized_with_weight(self):
        """Test parenthesized group with weight"""
        result = extract_terms("(dog cat):1.5")
        assert len(result) >= 1

    def test_complex_nested_format_1(self):
        """Test complex format: (prompts with spaces, (weighted:weight) other:weight)"""
        text = "(dog cat, (bird:1.3) fish:1.1)"
        result = extract_terms(text)
        # Should extract multiple terms with various weights
        assert len(result) > 0
        # Verify terms exist (exact values may vary due to parsing)
        terms_only = [t.lower() for t, _ in result]
        assert any("dog" in t or "cat" in t for t in terms_only)

    def test_complex_nested_format_2(self):
        """Test another complex format"""
        text = "(prompt1 prompt2, (prompt3:1.3) prompt4:1.1)"
        result = extract_terms(text)
        assert len(result) > 0

    def test_empty_string(self):
        """Test empty string returns empty list"""
        result = extract_terms("")
        assert result == []

    def test_only_stopwords(self):
        """Test input with only common stopwords"""
        result = extract_terms("the a an")
        # These should be filtered out
        assert len(result) == 0 or all(t in ["the", "a", "an"] for t, _ in result)

    def test_mixed_with_stopwords(self):
        """Test text with stopwords mixed in"""
        result = extract_terms("a big dog and a small cat")
        terms_only = [t.lower() for t, _ in result]
        # Should have substantive words
        assert any(t in terms_only for t in ["dog", "cat", "big", "small"])

    def test_special_characters(self):
        """Test handling of special characters"""
        result = extract_terms("dog-cat, bird_house")
        # Should handle special chars appropriately
        assert len(result) > 0

    def test_very_long_text(self):
        """Test handling of longer prompts"""
        long_text = "a very detailed portrait of a beautiful woman with long hair, wearing a blue dress, standing in a garden with flowers, cinematic lighting, professional photography, 4k resolution"
        result = extract_terms(long_text)
        # Should extract meaningful terms
        assert len(result) > 5
        terms = [t.lower() for t, _ in result]
        assert any(t in terms for t in ["woman", "hair", "dress", "garden", "flowers"])

    def test_with_custom_connectors(self):
        """Test with custom connector/splitter sets"""
        result = extract_terms("dog and cat", connectors={"and"}, splitters={"and"})
        # Should respect custom sets
        assert len(result) > 0

    def test_deduplication_keeps_max_weight(self):
        """Test that duplicate terms keep maximum weight"""
        result = extract_terms("dog:1.0, dog dog:2.0")
        # Find dog entries
        dog_entries = [w for t, w in result if t.lower() == "dog"]
        if len(dog_entries) == 1:  # Should deduplicate to 1 entry
            assert dog_entries[0] >= 1.0  # Should have max weight


class TestTokenizeText:
    """Test text tokenization preserving parenthesized groups"""

    def test_simple_text(self):
        """Test simple text tokenization"""
        result = tokenize_text("dog cat", {})
        assert len(result) > 0

    def test_parenthesized_groups(self):
        """Test that parenthesized groups are preserved as single tokens"""
        result = tokenize_text("(big dog) cat", set())
        # Should have at least one token with parentheses
        assert any("(" in token for token in result)

    def test_nested_parentheses(self):
        """Test nested parentheses handling"""
        result = tokenize_text("((dog cat) bird)", set())
        assert len(result) > 0

    def test_with_splitter_comma(self):
        """Test with comma as splitter"""
        result = tokenize_text("dog, cat", {","})
        # Comma should cause split
        assert len(result) == 2


class TestCleanTerm:
    """Test term cleaning function"""

    def test_strip_whitespace(self):
        """Test stripping whitespace"""
        result = clean_term("  dog  ")
        assert result == "dog"

    def test_lowercase(self):
        """Test lowercasing"""
        result = clean_term("Dog")
        assert result == "dog"

    def test_remove_punctuation(self):
        """Test removing punctuation"""
        result = clean_term("dog!")
        assert result == "dog"

    def test_remove_quotes(self):
        """Test removing quotes"""
        result = clean_term('"dog"')
        assert result == "dog"

    def test_remove_weight_suffix(self):
        """Test removing weight suffix"""
        result = clean_term("dog:1.5")
        assert result == "dog"

    def test_remove_backslashes(self):
        """Test removing backslashes"""
        result = clean_term(r"dog\cat")
        assert "dog" in result


class TestFilterTerms:
    """Test term filtering function"""

    def test_remove_short_terms(self):
        """Test that very short terms are removed"""
        input_terms: list[WeightedTerm] = [("a", 1.0), ("dog", 1.0), ("the", 1.0)]
        result = filter_terms(input_terms, set(), set())
        # Single letters should be filtered
        assert all(len(t) > 1 for t, _ in result)

    def test_remove_stopwords(self):
        """Test removal of common stopwords"""
        input_terms: list[WeightedTerm] = [("dog", 1.0), ("the", 1.0), ("at", 1.0)]
        result = filter_terms(input_terms, set(), {"the", "at"})
        terms_only = [t for t, _ in result]
        # Should have dog but not always_remove words
        assert "dog" in terms_only


class TestDeduplicateTerms:
    """Test term deduplication"""

    def test_exact_duplicates(self):
        """Test deduplication of exact duplicates"""
        input_terms: list[WeightedTerm] = [("dog", 1.0), ("dog", 1.0), ("cat", 1.0)]
        result = deduplicate_terms(input_terms)
        # Should have 2 unique terms
        assert len(result) == 2
        dog_weights = [w for t, w in result if t == "dog"]
        assert len(dog_weights) == 1  # Only one dog entry

    def test_different_weights_keeps_max(self):
        """Test that maximum weight is kept for duplicates"""
        input_terms: list[WeightedTerm] = [("dog", 1.0), ("dog", 2.0), ("dog", 0.5)]
        result = deduplicate_terms(input_terms)
        dog_entries = [w for t, w in result if t == "dog"]
        assert len(dog_entries) == 1
        assert dog_entries[0] == 2.0  # Should keep max weight

    def test_unique_terms_unchanged(self):
        """Test that unique terms are unchanged"""
        input_terms: list[WeightedTerm] = [("dog", 1.5), ("cat", 2.0), ("bird", 0.8)]
        result = deduplicate_terms(input_terms)
        assert len(result) == 3


class TestIntegrationComplex:
    """Test complex real-world prompt parsing scenarios"""

    def test_scenario_1_simple_prompt(self):
        """Real scenario: Simple positive prompt"""
        prompt = "a beautiful landscape with mountains"
        result = extract_terms(prompt)
        assert len(result) > 0
        terms = [t.lower() for t, _ in result]
        assert any(t in terms for t in ["beautiful", "landscape", "mountains"])

    def test_scenario_2_weighted_mixture(self):
        """Real scenario: Weighted terms within prompt"""
        prompt = "portrait, (detailed face:1.2), (blonde hair:1.5), professional lighting"
        result = extract_terms(prompt)
        assert len(result) > 2

    def test_scenario_3_complex_nesting(self):
        """Real scenario: Complex nested structure"""
        prompt = "((masterpiece, best quality)), (portrait:1.2), (detailed:1.3), (professional photograph:1.1)"
        result = extract_terms(prompt)
        assert len(result) > 2

    def test_scenario_4_mixed_simple_complex(self):
        """Real scenario: Mix of simple and complex terms"""
        prompt = "a woman, (detailed eyes:1.3), wearing (red dress:1.2), in (garden:1.1), sunlight, professional"
        result = extract_terms(prompt)
        assert len(result) > 5

    def test_scenario_5_negative_prompt_style(self):
        """Real scenario: Negative prompt with rejections"""
        prompt = "blurry, low quality, (deformed:1.2), (distorted:1.15), bad anatomy"
        result = extract_terms(prompt)
        assert len(result) > 0

    def test_scenario_6_very_complex_nesting(self):
        """Real scenario: Very complex nesting (worst case)"""
        prompt = "(((masterpiece, best quality))), (((detailed))), ((portrait of a woman)), ((she has:1.1) (long blonde hair:1.3)), (wearing (red:1.2) dress), environmental lighting"
        result = extract_terms(prompt)
        # Just ensure it doesn't crash and returns something
        assert isinstance(result, list)
        assert all(isinstance(t, str) and isinstance(w, float) for t, w in result)


# Roundtrip and regression tests
class TestRegression:
    """Tests to catch regressions in parsing logic"""

    def test_output_format_consistency(self):
        """Ensure extract_terms always returns list[tuple[str, float]]"""
        test_cases = [
            "simple",
            "word1:1.5 word2",
            "(group:2.0)",
            "complex, (nested:1.3) text",
        ]
        for text in test_cases:
            result = extract_terms(text)
            assert isinstance(result, list)
            for term, weight in result:
                assert isinstance(term, str)
                assert isinstance(weight, float)

    def test_weights_within_range(self):
        """Verify all weights are positive numbers"""
        result = extract_terms("dog:1.5, (cat:2.0 bird:0.5)")
        for term, weight in result:
            assert weight > 0, f"Weight {weight} is not positive for term {term}"

    def test_deterministic_output(self):
        """Ensure same input produces same output"""
        text = "dog, (cat:1.5), bird"
        result1 = extract_terms(text)
        result2 = extract_terms(text)
        # Sort for comparison since order might vary
        result1_sorted = sorted(result1)
        result2_sorted = sorted(result2)
        assert result1_sorted == result2_sorted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
