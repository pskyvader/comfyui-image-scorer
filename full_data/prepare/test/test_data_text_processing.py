import unittest
import inspect
import prepare.data.text_processing as shim
import text_data.text_processing as real

class TestDataTextProcessingShim(unittest.TestCase):
    def test_reexports(self):
        # Verify that the shim exports the same functions as the real module
        self.assertIs(shim.load_text_index, real.load_text_index)
        self.assertIs(shim.process_text_files, real.process_text_files)
        self.assertIs(shim.save_text_index, real.save_text_index)

if __name__ == '__main__':
    unittest.main()
