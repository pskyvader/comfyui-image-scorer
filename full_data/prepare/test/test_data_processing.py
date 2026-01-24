import unittest
from unittest.mock import patch, MagicMock, Mock
from prepare.data.processing import collect_valid_files, load_existing_data, remove_existing_outputs
from shared.config import config

class TestDataProcessing(unittest.TestCase):
    
    @patch('prepare.data.processing.load_index')
    @patch('prepare.data.processing.load_json_list_robust')
    def test_load_existing_data(self, mock_load_json, mock_load_index):
        config["index_file"] = "index.json"
        config["vectors_file"] = "vectors.jsonl"
        config["scores_file"] = "scores.jsonl"
        
        mock_load_index.return_value = ["file1.png", "file2.png"]
        # Mock vectors as simple lists
        mock_load_json.side_effect = [
            [[0.1, 0.2], [0.3, 0.4]], # vectors
            [1.0, 2.0]  # scores
        ]
        
        index, vectors, scores = load_existing_data()
        
        self.assertEqual(index, ["file1.png", "file2.png"])
        self.assertEqual(vectors, [[0.1, 0.2], [0.3, 0.4]])
        self.assertEqual(scores, [1.0, 2.0])

    @patch('prepare.data.processing.clean_training_artifacts')
    @patch('prepare.data.processing.os.remove')
    @patch('prepare.data.processing.os.path.exists')
    def test_remove_existing_outputs(self, mock_exists, mock_remove, mock_clean):
        mock_exists.return_value = True
        remove_existing_outputs()
        self.assertTrue(mock_remove.called)
        mock_clean.assert_called_once()

    @patch('prepare.data.processing.os.remove')
    @patch('prepare.data.processing.os.path.exists')
    def test_clean_training_artifacts(self, mock_exists, mock_remove):
        from prepare.data.processing import clean_training_artifacts
        mock_exists.return_value = True
        clean_training_artifacts()
        # Should call remove for each file in the list (6 files)
        self.assertEqual(mock_remove.call_count, 6)


    @patch('prepare.data.processing.load_metadata_entry')
    @patch('prepare.data.processing.Image.open')
    @patch('prepare.data.processing.config') # To mock image_root access indirectly if needed, but config is global
    @patch('prepare.data.processing.os.path.relpath')
    def test_collect_valid_files(self, mock_relpath, mock_config, mock_image_open, mock_load_meta):
        # Setup
        config["image_root"] = "/images"
        mock_relpath.side_effect = lambda p, start: p.split("/")[-1] # Simple mock: /images/file1.png -> file1.png
        files = [
            ("/images/file1.png", "/images/file1.json"),
            ("/images/file2.png", "/images/file2.json")
        ]
        processed_files = {"file2.png"} # file2 is already processed
        error_log = []
        
        # Mocks
        # file1: valid
        mock_load_meta.return_value = ({"score": 5}, "20230101", None)
        
        mock_img = Mock()
        mock_img.size = (512, 512)
        mock_img_ctx = Mock()
        mock_img_ctx.__enter__ = Mock(return_value=mock_img)
        mock_img_ctx.__exit__ = Mock(return_value=None)
        mock_image_open.return_value = mock_img_ctx
        
        # Execution
        result = collect_valid_files(files, processed_files, error_log)
        
        # Assertions
        # Should only contain file1, because file2 is in processed_files
        self.assertEqual(len(result), 1)
        path, entry, ts, fid = result[0]
        self.assertEqual(fid, "file1.png")
        self.assertEqual(entry["width"], 512)
        self.assertEqual(ts, "20230101")
        
    @patch('prepare.data.processing.load_metadata_entry')
    def test_collect_valid_files_bad_score(self, mock_load_meta):
        # Setup
        config["image_root"] = "/images"
        files = [("/images/file1.png", "/images/file1.json")]
        processed_files = set()
        error_log = []
        
        # Mock load returning None (bad json)
        mock_load_meta.return_value = (None, None, "bad_json")
        
        result = collect_valid_files(files, processed_files, error_log)
        
        self.assertEqual(len(result), 0)
        self.assertEqual(len(error_log), 1)
        self.assertEqual(error_log[0]["reason"], "bad_json")

    def test_overflow_pads_current_vector(self):
        from unittest.mock import patch
        from prepare.data.processing import process_and_append_data
        from prepare.config.schema import get_feature_vector_length, IMAGE_VEC_LEN

        # Create fake existing vectors_list with one vector of old expected length
        slots = {
            "cfg": 1,
            "steps": 1,
            "lora_weight": 1,
            "steps_cfg": 1,
            "lora": 5,
            "sampler": 3,
            "scheduler": 3,
            "model": 4,
            "width": 1,
            "height": 1,
            "aspect_ratio": 1,
            "negative_terms": 1,
            "positive_terms": 1,
        }
        mode = "embedding"
        dim = 768
        feature_len = get_feature_vector_length(slots, mode, dim)
        image_vec = [0.0] * IMAGE_VEC_LEN
        existing_full = image_vec + [0.0] * feature_len
        vectors_list = [existing_full]
        scores_list = [0.0]
        index_list = []
        processed_files = set()

        # Simulate collected_data containing one item
        collected_data = [("/images/test.png", {"score": 1.0, "lora": "new"}, "ts", "test.png")]
        image_vectors = [image_vec]

        # Patch build_feature_vector to simulate overflow and smaller feature vector
        small_feature = [0.0] * (feature_len - 1)
        def fake_build_feature_vector(*args, **kwargs):
            return (small_feature.copy(), slots, ["lora"])  # overflow

        with patch("prepare.data.processing.build_feature_vector", side_effect=fake_build_feature_vector):
            with patch("prepare.data.processing.pad_existing_vectors"):
                # Ensure vector order is present by patching the loader for get_vector_order()
                with patch('prepare.config.schema.load_vector_schema', return_value={
                    "slots": slots,
                    "order": [
                        "cfg",
                        "steps",
                        "lora_weight",
                        "steps_cfg",
                        "lora",
                        "sampler",
                        "scheduler",
                        "model",
                        "width",
                        "height",
                        "aspect_ratio",
                        "negative_terms",
                        "positive_terms",
                    ],
                }):
                    process_and_append_data(
                        collected_data,
                        image_vectors,
                        vectors_list,
                        scores_list,
                        index_list,
                        processed_files,
                        {"slots": slots},
                    )

        # After processing, all vectors should have same length
        lengths = set(len(v) for v in vectors_list)
        assert len(lengths) == 1, f"Vectors have mismatched lengths: {lengths}"

if __name__ == '__main__':
    unittest.main()
