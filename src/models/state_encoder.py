import torch
import torch.nn as nn
from typing import Tuple

class StateEncoder(nn.Module):
    """
    MLP state encoder that transforms raw features into a dense latent representation.
    Also includes a decoder to reconstruct the original state from the latent representation.
    """
    def __init__(self, input_dim: int = 76, hidden: int = 192, latent_dim: int = 128, dropout: float = 0.15):
        super(StateEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden = hidden
        self.latent_dim = latent_dim
        
        # Encoder MLP: Linear -> LayerNorm -> GELU -> Dropout -> Linear -> LayerNorm
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        
        # Decoder MLP: Linear -> LayerNorm -> GELU -> Dropout -> Linear -> LayerNorm
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, input_dim),
            nn.LayerNorm(input_dim)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode raw features to latent representation.
        
        Args:
            x (torch.Tensor): Raw input features of shape (..., input_dim)
            
        Returns:
            torch.Tensor: Latent representation of shape (..., latent_dim)
        """
        return self.encoder(x)
        
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to reconstruct state features.
        
        Args:
            z (torch.Tensor): Latent representation of shape (..., latent_dim)
            
        Returns:
            torch.Tensor: Reconstructed features of shape (..., input_dim)
        """
        return self.decoder(z)
