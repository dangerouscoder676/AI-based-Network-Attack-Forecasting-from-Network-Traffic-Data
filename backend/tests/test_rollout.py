import sys
import os
import unittest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.world_model import WorldModel
from src.prediction.rollout_engine import RolloutEngine
from src.utils.config import load_config

class TestRolloutEngine(unittest.TestCase):
    def setUp(self):
        self.config = load_config('config.yaml')
        self.model = WorldModel.from_config(self.config)
        self.model.eval()
        self.engine = RolloutEngine(self.model)
        self.feature_dim = self.config['features']['num_state_features']
        self.seq_len = 16
        self.context = torch.randn(1, self.seq_len, self.feature_dim)

    def test_rollout(self):
        k_steps = 5
        results = self.engine.rollout(self.context, k_steps=k_steps)
        self.assertEqual(len(results['probabilities']), k_steps)
        self.assertEqual(len(results['stages']), k_steps)
        self.assertEqual(results['predicted_states'].shape, (k_steps, self.feature_dim))

if __name__ == '__main__':
    unittest.main()
