import unittest
from unittest.mock import patch, MagicMock
import sys
import importlib.util
import os
import numpy as np

# Import helper
def import_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "lib", "scorer.py"))
scorer_mod = import_module_from_path("scorer", lib_path)

class TestScorerLib(unittest.TestCase):

    @patch('scorer.ort.InferenceSession')
    @patch('scorer.np.load')
    @patch('scorer.Path') # We mock Path to control exists() checks
    def test_init_success(self, mock_path_cls, mock_load, mock_ort):
        # Setup filesystem mocks
        mock_dir = MagicMock()
        mock_path_cls.return_value = mock_dir
        mock_dir.__truediv__.return_value = mock_dir # Simplified: any path is mock_dir
        mock_dir.exists.return_value = True
        
        # Setup numpy load
        mock_npz = MagicMock()
        mock_npz.__getitem__.side_effect = lambda key: np.array([0, 1]) if key in ["kept_indices", "interaction_indices"] else None
        mock_load.return_value = mock_npz
        
        # Setup ONNX
        mock_sess = MagicMock()
        mock_sess.get_inputs.return_value = [MagicMock(name="input")]
        mock_ort.return_value = mock_sess
        
        scorer = scorer_mod.ScorerModel("dummy_dist")
        
        self.assertIsNotNone(scorer.sess)
        self.assertIsNotNone(scorer.kept_indices)

    def test_predict_mocked(self):
        # Need to partially init or mock the whole class
        pass

if __name__ == '__main__':
    unittest.main()
