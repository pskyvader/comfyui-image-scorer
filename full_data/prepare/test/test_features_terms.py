import unittest
from prepare.features.terms import extract_weight_from_paren, split_weighted_compound

class TestFeaturesTerms(unittest.TestCase):
    def test_extract_weight_from_paren(self):
        self.assertEqual(extract_weight_from_paren('(foo:1.2)'), ('foo', 1.2))
        self.assertEqual(extract_weight_from_paren('(foo)'), ('foo', 1.0))
        self.assertEqual(extract_weight_from_paren('foo'), ('foo', 1.0))

    def test_split_weighted_compound(self):
        res = split_weighted_compound('apple, banana', 1.0)
        self.assertEqual(res, [('apple', 1.0), ('banana', 1.0)])
        
        res = split_weighted_compound('(apple:1.5), banana', 1.0)
        # (apple:1.5) -> apple with 1.5
        # banana -> banana with 1.0
        self.assertEqual(res, [('apple', 1.5), ('banana', 1.0)])

if __name__ == '__main__':
    unittest.main()
