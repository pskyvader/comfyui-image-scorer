import unittest
from unittest.mock import patch
from prepare.features.assemble import embedding_component, get_component_vector
from shared.config import config

class TestFeaturesAssemble(unittest.TestCase):
    def test_embedding_component_default(self):
        config["prompt_representation"] = {'mode': 'default'}
        with patch('prepare.features.assemble.weighted_presence') as mock_wp:
            mock_wp.return_value = [1.0, 0.0]
            result = embedding_component([(0, 1.0)], 2)
            self.assertEqual(result, [1.0, 0.0])

    def test_embedding_component_embedding(self):
        config["prompt_representation"] = {'mode': 'embedding', 'dim': 2}
        # indices are (index, weight). 
        # The code does: emb = np.asarray(indices, dtype=np.float32).flatten()
        # If indices = [(0, 1.0)], emb = [0.0, 1.0]
        # vec = emb[:2] -> [0.0, 1.0]
        result = embedding_component([(0, 1.0)], 2)
        self.assertEqual(result, [0.0, 1.0])

    @patch('prepare.features.assemble.get_slot_size')
    def test_get_component_vector_cfg(self, mock_size):
        mock_size.return_value = 1
        # Added width, height, aspect_ratio args (0.0, 0.0, 0.0)
        res = get_component_vector('cfg', 0.5, 0.0, 0.0, 0, 0, 0, 0, 0.0, 0.0, 0.0, [], [], {}, "default", 0)
        self.assertEqual(res, [0.5])

if __name__ == '__main__':
    unittest.main()
