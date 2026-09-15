import sys
import os
import unittest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.world_model import WorldModel
from src.utils.config import load_config

class TestWorldModel(unittest.TestCase):
    def setUp(self):
        self.config = load_config('config.yaml')
        self.model = WorldModel.from_config(self.config)
        self.batch_size = 4
        self.seq_len = 16
        self.feature_dim = self.config['features']['num_state_features']
        self.x = torch.randn(self.batch_size, self.seq_len, self.feature_dim)

    def test_forward_pass(self):
        out = self.model(self.x)
        self.assertIn('next_state_pred', out)
        self.assertIn('infiltration_prob', out)
        self.assertIn('stage_logits', out)
        self.assertEqual(out['next_state_pred'].shape, (self.batch_size, self.feature_dim))
        self.assertEqual(out['infiltration_prob'].shape, (self.batch_size, 1))

    def test_loss_computation(self):
        out = self.model(self.x)
        target_next = torch.randn(self.batch_size, self.feature_dim)
        target_infil = torch.tensor([0.0, 1.0, 0.0, 1.0]).unsqueeze(1)
        target_stage = torch.tensor([0, 2, 0, 4])
        
        targets = {
            'next_state': target_next,
            'infiltration_label': target_infil,
            'stage_label': target_stage
        }
        
        losses = self.model.compute_loss(out, targets)
        self.assertIn('total_loss', losses)
        self.assertTrue(torch.isfinite(losses['total_loss']))

if __name__ == '__main__':
    unittest.main()
