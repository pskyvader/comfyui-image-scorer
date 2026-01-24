import unittest
import os

class TestTextTrainingPlaceholder(unittest.TestCase):
    def test_placeholder_exists(self):
        self.assertTrue(os.path.exists(os.path.join(os.path.dirname(__file__), "..", "text", "train_text_model.py")))

if __name__ == '__main__':
    unittest.main()
