import unittest
from unittest.mock import patch, MagicMock
from prepare.features.embeddings import load_model, encode_prompt
import numpy as np

class TestEmbeddings(unittest.TestCase):

    @patch('prepare.features.embeddings.SentenceTransformer')
    def test_load_model(self, mock_st):
        # Setup mock
        mock_instance = MagicMock()
        mock_instance.get_sentence_embedding_dimension.return_value = 384
        mock_st.return_value = mock_instance
        
        # Test success
        model = load_model("test-model", 384, "cpu")
        self.assertEqual(model, mock_instance)
        
        # Test dim mismatch
        with self.assertRaises(ValueError):
            load_model("test-model", 512, "cpu")

    def test_encode_prompt(self):
        mock_model = MagicMock()
        mock_model.encode.return_value = [0.1, 0.2]
        
        res = encode_prompt("hello", mock_model)
        
        self.assertTrue(isinstance(res, np.ndarray))
        self.assertEqual(res.dtype, np.float32)
        mock_model.encode.assert_called_with("hello")

if __name__ == '__main__':
    unittest.main()
