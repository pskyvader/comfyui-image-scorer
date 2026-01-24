import unittest
from unittest.mock import patch
from prepare.features.core import build_feature_vector

class TestFeaturesCore(unittest.TestCase):

    @patch('prepare.features.core.parse_entry_meta')
    @patch('prepare.features.core.get_categorical_indices')
    @patch('prepare.features.core.extract_terms')
    @patch('prepare.features.core.map_terms_to_indices')
    @patch('prepare.features.core.assemble_feature_vector')
    def test_build_feature_vector_default(self, mock_assemble, mock_map, mock_extract, mock_cat, mock_parse):
        # Mocks
        mock_parse.return_value = (0.5, 0.5, 1.0, "s", "sc", "m", "l", 0.5, 0.5, 1.0)
        mock_cat.return_value = ((0,0,0,0), ("known", "known", "known", "known"))
        mock_extract.side_effect = [{"term1"}, {"term2"}]
        mock_map.return_value = ([], []) # indices, statuses
        mock_assemble.return_value = ([1.0, 2.0], [])
        
        entry = {}
        slots = {"positive_terms": 10, "negative_terms": 10}
        mode = "default"
        dim = 0
        normalization = {}
        order = []
        
        vec, statuses, debug = build_feature_vector(entry, slots, mode, dim, normalization, order)
        
        self.assertEqual(vec, [1.0, 2.0])
        self.assertEqual(statuses, slots)

if __name__ == '__main__':
    unittest.main()
