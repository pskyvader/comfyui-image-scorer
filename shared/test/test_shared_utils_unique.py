import unittest
from shared.utils import parse_custom_text, first_present, one_hot, binary_presence, weighted_presence

class TestSharedUtils(unittest.TestCase):
    
    def test_parse_custom_text(self):
        self.assertEqual(parse_custom_text(), {})
        self.assertEqual(parse_custom_text(None), {})
        self.assertEqual(parse_custom_text(""), {})
        self.assertEqual(parse_custom_text("{'a': 1}"), {'a': 1})
        self.assertEqual(parse_custom_text({'a': 1}), {'a': 1})
        self.assertEqual(parse_custom_text("invalid"), {})
    
    def test_first_present(self):
        d = {'a': 1, 'b': None, 'c': 3}
        self.assertEqual(first_present(d, ('a', 'b'), 0), 1)
        self.assertEqual(first_present(d, ('b', 'c'), 0), 3)
        self.assertEqual(first_present(d, ('x', 'y'), 0), 0)
    
    def test_one_hot(self):
        self.assertEqual(one_hot(0, 3), [1, 0, 0])
        self.assertEqual(one_hot(2, 3), [0, 0, 1])
        self.assertEqual(one_hot(3, 3), [0, 0, 0]) # Out of bounds
    
    def test_binary_presence(self):
        self.assertEqual(binary_presence([0, 2], 3), [1, 0, 1])
        self.assertEqual(binary_presence([], 3), [0, 0, 0])
    
    def test_weighted_presence(self):
        # Case 1: list of tuples
        self.assertEqual(weighted_presence([(0, 0.5), (2, 0.8)], 3), [0.5, 0.0, 0.8])
        # Case 2: indices, weights, length
        self.assertEqual(weighted_presence([0, 2], [0.5, 0.8], 3), [0.5, 0.0, 0.8])
        # Max behavior
        self.assertEqual(weighted_presence([(0, 0.5), (0, 0.9)], 3), [0.9, 0.0, 0.0])

if __name__ == '__main__':
    unittest.main()
