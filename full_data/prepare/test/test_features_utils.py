import unittest
import prepare.features.utils as shim
import shared.utils as real

class TestFeaturesUtils(unittest.TestCase):
    def test_reexports(self):
        self.assertIs(shim.one_hot, real.one_hot)
        self.assertIs(shim.binary_presence, real.binary_presence)
        self.assertIs(shim.weighted_presence, real.weighted_presence)

if __name__ == '__main__':
    unittest.main()
