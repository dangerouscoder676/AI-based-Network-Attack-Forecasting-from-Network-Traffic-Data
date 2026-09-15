import torch
import torch.nn as nn
from typing import Dict, Tuple

class LSTMDynamics(nn.Module):
    """
    LSTM comparison model for sequence modeling.
    """
    def __init__(self, d_model: int = 128, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.15):
        super(LSTMDynamics, self).__init__()
        
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self._init_weights()
        
    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Forget gate bias initialization
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)
                
        nn.init.ones_(self.layer_norm.weight)
        nn.init.zeros_(self.layer_norm.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            context: Tensor of shape (batch_size, hidden_dim) representing the last time step.
            attn_weights: Empty dictionary (for compatibility with TemporalTransformer).
        """
        output, (hn, cn) = self.lstm(x)
        
        # Take the output of the last time step
        last_step_output = output[:, -1, :]
        last_step_output = self.layer_norm(last_step_output)
        
        # No attention weights for LSTM
        return last_step_output, {}
