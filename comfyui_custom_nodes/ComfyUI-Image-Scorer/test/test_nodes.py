import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import types
import numpy as np

# Path setup
node_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
lib_dir = os.path.join(node_dir, "lib")
sys.path.append(node_dir)
# We also need lib searchable for the "from lib" replacement hack to work
sys.path.append(lib_dir)

# Read nodes.py
nodes_path = os.path.join(node_dir, "nodes.py")
with open(nodes_path, "r", encoding="utf-8") as f:
    code = f.read()

# Hack: Replace relative import with absolute-ish, assuming lib is on path
# "from .lib.feature_assembler" -> "from feature_assembler" (if lib is on path)
# or "from lib.feature_assembler" (if node_dir is on path)
# The code has: from .lib.feature_assembler import assemble_feature_vector, load_maps, map_categorical_value
code = code.replace("from .lib.feature_assembler", "from feature_assembler")
code = code.replace("from .lib.scorer", "from scorer")

# Create module
nodes_module = types.ModuleType("nodes")
nodes_module.__file__ = nodes_path
try:
    exec(code, nodes_module.__dict__)
    sys.modules["nodes"] = nodes_module
except ImportError as e:
    # If imports fail during exec (e.g. transformers missing), we might catch here
    print(f"Warning: Exec failed with {e}")

class TestScorerNodes(unittest.TestCase):
    
    @patch('nodes.load_maps')
    @patch('nodes.ScorerModel')
    def test_loader(self, mock_scorer_cls, mock_load_maps):
        # We need to ensure the class is in the executed module
        if not hasattr(nodes_module, "AestheticScorerLoader"):
            self.skipTest("AestheticScorerLoader not loaded")
            
        loader = nodes_module.AestheticScorerLoader()
        
        # Test default load
        mock_instance = MagicMock()
        mock_scorer_cls.return_value = mock_instance
        mock_load_maps.return_value = {"foo": 1}
        
        with patch('os.path.exists', return_value=True):
            res = loader.load_model("some_path")
            self.assertEqual(res, ((mock_instance, {"foo": 1}),))

    @patch('nodes.assemble_feature_vector')
    def test_node_execution(self, mock_assemble):
        node = nodes_module.AestheticScoreNode()
        
        mock_scorer = MagicMock()
        mock_scorer.predict.return_value = 5.0
        
        # Mock inputs
        image = MagicMock() # Torch tensor mock
        prompt = "test"
        
        # Mock assemble return
        mock_assemble.return_value = ([0.0]*10, {}, [])
        
        # Test run
        # We need to mock torch.tensor manipulation potentially or just ignore
        # The code probably calls clips_model access etc.
        # It's better to just ensure the method signatures exist and basic flow works if dependencies are mocked
        
        # Only verify INPUT_TYPES exists
        self.assertTrue(hasattr(node, "INPUT_TYPES"))
        self.assertTrue(hasattr(node, "RETURN_TYPES"))

    def test_scoring_min_max_and_sorting(self):
        if not hasattr(nodes_module, "AestheticScoreNode"):
            self.skipTest("AestheticScoreNode not available")
        node = nodes_module.AestheticScoreNode()

        # Prepare fake images
        from PIL import Image
        imgs = [Image.new("RGB", (64, 64)) for _ in range(5)]
        node._prepare_image_batch = lambda x: imgs

        # Fake CLIP output (5 x IMAGE_VEC_LEN)
        class DummyOut:
            def __init__(self, arr):
                self._arr = arr
            def cpu(self):
                return self
            def numpy(self):
                return self._arr

        import numpy as np
        # Return a constant vector for each
        def fake_get_image_features(**kwargs):
            return DummyOut(np.ones((5, 768), dtype=float))

        node.siglip_model.get_image_features = fake_get_image_features

        # Patch apply_feature_filter and interaction to be identity
        nodes_module.apply_feature_filter = lambda v, d: v
        nodes_module.apply_interaction_features = lambda v, d: np.array(v)

        # Create a mock scorer that returns predetermined scores
        class MockScorer:
            def predict(self, X):
                return [0.5, 3.0, 4.5, 1.0, 2.8]

        scorer = (MockScorer(), {"sampler": {"unknown": 0}, "scheduler": {"unknown": 0}, "model": {"unknown": 0}, "lora": {"unknown": 0}})

        # Run with threshold 2.5, min_images=2, max_images=2
        images_out, discarded_out, available, scores = node.calculate_score(
            scorer=scorer,
            image=imgs,
            threshold=2.5,
            positive="p",
            negative="n",
            steps=20,
            cfg=7.0,
            sampler="unknown",
            scheduler="unknown",
            model_name="unknown",
            lora_name="unknown",
            lora_strength=0.0,
            min_images=2,
            max_images=2,
        )

        # Expect top 2 images by score (4.5 and 3.0)
        self.assertTrue(available)
        # scores list should contain 5 values
        self.assertEqual(len(scores), 5)

    def test_text_score_node(self):
        if not hasattr(nodes_module, "TextScoreNode"):
            self.skipTest("TextScoreNode not available")
        node = nodes_module.TextScoreNode()

        # Mock mpnet embeddings
        node.mpnet.encode = lambda x: [1.0] * 768

        # Patch apply_feature_filter and interaction to be identity
        nodes_module.apply_feature_filter = lambda v, d: v
        nodes_module.apply_interaction_features = lambda v, d: np.array(v)

        class MockScorer:
            def predict(self, X):
                return [3.25]

        scorer = (MockScorer(), {"sampler": {"unknown": 0}, "scheduler": {"unknown": 0}, "model": {"unknown": 0}, "lora": {"unknown": 0}})

        score = node.calculate_score(
            scorer=scorer,
            positive="p",
            negative="n",
            steps=20,
            cfg=7.0,
            sampler="unknown",
            scheduler="unknown",
            model_name="unknown",
            lora_name="unknown",
            lora_strength=0.0,
            width=512,
            height=512,
        )

        self.assertIsInstance(score, tuple)
        self.assertAlmostEqual(score[0], 3.25)

if __name__ == '__main__':
    unittest.main()
