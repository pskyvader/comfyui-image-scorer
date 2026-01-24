import unittest
from unittest.mock import patch, MagicMock, mock_open
import os
from prepare.data.metadata import write_error_log, load_metadata_entry, extract_text_components
from shared.config import config

class TestMetadata(unittest.TestCase):

    @patch('prepare.data.metadata.atomic_write_json')
    @patch('prepare.data.metadata.os.path.exists')
    @patch('prepare.data.metadata.os.remove')
    def test_write_error_log_with_errors(self, mock_remove, mock_exists, mock_write):
        error_log = [{"file": "test.txt", "error": "failed"}]
        write_error_log(error_log, "path/to/log.json")
        mock_write.assert_called_once_with("path/to/log.json", error_log, indent=2)
        mock_remove.assert_not_called()

    @patch('prepare.data.metadata.atomic_write_json')
    @patch('prepare.data.metadata.os.path.exists')
    @patch('prepare.data.metadata.os.remove')
    def test_write_error_log_empty(self, mock_remove, mock_exists, mock_write):
        error_log = []
        mock_exists.return_value = True
        write_error_log(error_log, "path/to/log.json")
        mock_write.assert_not_called()
        mock_remove.assert_called_once_with("path/to/log.json")

    @patch('prepare.data.metadata.load_single_entry_mapping')
    def test_load_metadata_entry_success(self, mock_load):
        mock_load.return_value = ({"foo": "bar"}, "timestamp", None)
        payload, ts, err = load_metadata_entry("test.json")
        self.assertEqual(payload, {"foo": "bar"})
        self.assertEqual(ts, "timestamp")
        self.assertIsNone(err)

    @patch('prepare.data.metadata.load_single_entry_mapping')
    def test_load_metadata_entry_not_found(self, mock_load):
        mock_load.return_value = (None, None, "not_found")
        payload, ts, err = load_metadata_entry("test.json")
        self.assertIsNone(payload)
        self.assertEqual(err, "json_not_found")

    @patch('prepare.data.metadata.parse_entry_meta')
    @patch('prepare.data.metadata.extract_terms')
    def test_extract_text_components(self, mock_extract, mock_parse):
        # Mock parse_entry_meta return values
        mock_parse.return_value = (
            7.0, # cfg
            20, # steps
            1.0, # lora_weight
            0, # sampler
            0, # scheduler
            0, # model
            0, # lora
            512, # width
            512, # height
            1.0 # ar
        )
        mock_extract.side_effect = [{"term1", "term2"}, {"term3"}]
        
        entry = {"positive_prompt": "hello", "negative_prompt": "world"}
        result = extract_text_components(entry)
        
        self.assertIn("positive_terms", result)
        self.assertIn("negative_terms", result)
        self.assertEqual(result["positive_terms"], {"term1", "term2"})

if __name__ == '__main__':
    unittest.main()
