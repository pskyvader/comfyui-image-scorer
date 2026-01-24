import unittest
from unittest.mock import patch
from prepare.features.meta import parse_entry_meta, parse_custom_text

class TestFeaturesMeta(unittest.TestCase):
    def test_parse_custom_text(self):
        self.assertEqual(parse_custom_text(None), {})
        self.assertEqual(parse_custom_text({'a': 1}), {'a': 1})
        self.assertEqual(parse_custom_text("{'a': 1}"), {'a': 1})
        self.assertEqual(parse_custom_text("invalid"), {})

    def test_parse_entry_meta(self):
        normalization = {'cfg_max': 10.0, 'steps_max': 100.0}
        entry = {
            'cfg': 5.0,
            'steps': 50,
            'lora_weight': 0.5,
            'sampler': 'euler',
            'scheduler': 'normal',
            'model': 'sd15',
            'lora': 'l1',
            'width': 512,
            'height': 512,
            'aspect_ratio': 1.0
        }
        res = parse_entry_meta(entry, normalization)
        # cfg_norm = 5/10 = 0.5
        # steps_norm = 50/100 = 0.5
        self.assertEqual(res[0], 0.5)
        self.assertEqual(res[1], 0.5)
        self.assertEqual(res[2], 0.5)
        self.assertEqual(res[3], 'euler')
        self.assertEqual(res[4], 'normal')
        self.assertEqual(res[5], 'sd15')
        self.assertEqual(res[6], 'l1')

if __name__ == '__main__':
    unittest.main()
