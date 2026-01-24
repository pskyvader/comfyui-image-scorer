import unittest
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

lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "lib", "feature_assembler.py"))
assembler = import_module_from_path("feature_assembler", lib_path)

class TestFeatureAssembler(unittest.TestCase):
    
    def test_one_hot(self):
        self.assertEqual(assembler.one_hot(0, 3), [1.0, 0.0, 0.0])
        self.assertEqual(assembler.one_hot(2, 3), [0.0, 0.0, 1.0])
        self.assertEqual(assembler.one_hot(-1, 3), [0.0, 0.0, 0.0])
    
    def test_assemble_feature_vector(self):
        meta = {
            "cfg": 7.0,
            "steps": 20,
            "lora_weight": 1.0,
            "width": 512,
            "height": 512,
            "aspect_ratio": 1.0
        }
        # Fake embeddings
        pos_emb = np.ones(768, dtype=np.float32)
        neg_emb = np.zeros(768, dtype=np.float32)
        
        cat_indices = {
            "sampler": 0,
            "scheduler": 0,
            "model": 0,
            "lora": 0
        }
        
        vec = assembler.assemble_feature_vector(meta, pos_emb, neg_emb, cat_indices)
        
        self.assertTrue(isinstance(vec, np.ndarray))
        # Check exact length by summing configured slot sizes
        expected_len = 0
        for k in assembler.SCHEMA_ORDER:
            expected_len += assembler.get_slot_size(k)

        self.assertEqual(len(vec), expected_len)

    def test_dynamic_config_load(self):
        # Attempt to autodiscover and load prepare_config.json from repo
        path = assembler.load_prepare_config()
        # It should find the repo's prepare_config.json during tests
        self.assertIsNotNone(path)
        # After loading, schema order and embedding dim should reflect config
        self.assertTrue(len(assembler._SCHEMA_ORDER) >= 1)
        self.assertTrue(assembler._EMBEDDING_DIM > 0)

    def test_load_prepare_config_strict_path(self):
        # Passing a non-existent explicit path should raise FileNotFoundError
        import os
        missing = os.path.join(os.path.dirname(__file__), "nonexistent_prepare_config.json")
        with self.assertRaises(FileNotFoundError):
            assembler.load_prepare_config(missing)


if __name__ == '__main__':
    unittest.main()
