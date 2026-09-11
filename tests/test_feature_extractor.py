import sys
import os
import unittest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.feature_extractor import FeatureExtractor
from src.data.synthetic_generator import SyntheticGenerator
from src.utils.config import load_config

class TestFeatureExtractor(unittest.TestCase):
    def setUp(self):
        self.config = load_config('config.yaml')
        self.extractor = FeatureExtractor(self.config['features']['flow_features'])
        generator = SyntheticGenerator()
        df, _, self.feature_names = generator.generate(num_windows=10)
        self.df = pd.DataFrame(df, columns=self.feature_names)

    def test_extract(self):
        extracted = self.extractor.extract(self.df)
        self.assertIsNotNone(extracted)
        self.assertGreaterEqual(extracted.shape[1], 68)

    def test_feature_names(self):
        names = self.extractor.get_feature_names()
        self.assertTrue(len(names) > 0)
        self.assertIn("syn_to_total_ratio", names)

if __name__ == '__main__':
    unittest.main()
