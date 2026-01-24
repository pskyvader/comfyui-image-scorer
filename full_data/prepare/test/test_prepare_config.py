import unittest
from prepare.config.manager import load_vector_schema, save_vector_schema
from shared.config import config

class TestConfigManager(unittest.TestCase):
    def test_load_save_vector_schema(self):
        # Setup
        if "prepare" not in config:
            config["prepare"] = {}
        original_schema = config["prepare"].get("vector_schema")
        
        test_schema = {"slots": {"test": 1}}
        
        # Test Save
        save_vector_schema(test_schema)
        self.assertEqual(config["prepare"]["vector_schema"], test_schema)
        
        # Test Load
        loaded = load_vector_schema()
        self.assertEqual(loaded, test_schema)
        
        # Cleanup
        if original_schema:
            config["prepare"]["vector_schema"] = original_schema

if __name__ == '__main__':
    unittest.main()
