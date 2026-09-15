import torch
import torch.nn as nn
from typing import Tuple

class InfiltrationHead(nn.Module):
    """
    Prediction heads for infiltration probability and stage classification.
    """
    def __init__(self, latent_dim: int = 128, num_stages: int = 6):
        super(InfiltrationHead, self).__init__()
        
        # Binary infiltration classification: latent_dim -> 64 -> 1
        self.infiltration_mlp = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Stage classification (multiclass): latent_dim -> 64 -> num_stages
        self.stage_mlp = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_stages)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Latent context vector of shape (batch_size, latent_dim)
            
        Returns:
            Tuple containing:
            - infiltration_prob: Tensor of shape (batch_size, 1) with probabilities in [0, 1]
            - stage_logits: Tensor of shape (batch_size, num_stages) with raw logits
        """
        infiltration_logits = self.infiltration_mlp(x)
        infiltration_prob = torch.sigmoid(infiltration_logits)
        
        stage_logits = self.stage_mlp(x)
        
        return infiltration_prob, stage_logits
