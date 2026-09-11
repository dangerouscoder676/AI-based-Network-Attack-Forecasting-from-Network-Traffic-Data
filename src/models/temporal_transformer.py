import math
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return x

class TemporalTransformer(nn.Module):
    """
    Transformer for learning P(S_{t+1} | S_t) with causal masking and attention weight extraction.
    """
    def __init__(self, d_model: int = 128, nhead: int = 4, num_layers: int = 3, 
                 dim_feedforward: int = 256, dropout: float = 0.15):
        super(TemporalTransformer, self).__init__()
        
        self.d_model = d_model
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            norm_first=True,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attention_weights: Dict[int, torch.Tensor] = {}
        
        self._register_attention_hooks()
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def _register_attention_hooks(self):
        """
        Registers forward hooks on the MultiheadAttention layers to capture attention weights.
        """
        def get_hook(layer_idx):
            def hook(module, input, output):
                # multihead_attention forward returns (attn_output, attn_weights)
                # where attn_weights is of shape (batch, nhead, seq_len, seq_len) if average_attn_weights=False
                # Note: PyTorch's default returns (batch, seq_len, seq_len) averaged over heads unless average_attn_weights=False.
                # Since we can't easily change the TransformerEncoderLayer's MHA call, we extract what is given.
                # Standard hook on TransformerEncoderLayer MHA:
                attn_output, attn_weights = output
                self.attention_weights[layer_idx] = attn_weights
            return hook
            
        for i, layer in enumerate(self.transformer_encoder.layers):
            # register hook on self_attn
            layer.self_attn.register_forward_hook(get_hook(i))

    def _generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """
        Generates a causal mask.
        """
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            context: Tensor of shape (batch_size, d_model) representing the last time step.
            attn_weights: Dictionary of attention weights from each layer.
        """
        # Clear previous attention weights
        self.attention_weights.clear()
        
        seq_len = x.size(1)
        # Apply positional encoding
        x = self.pos_encoder(x)
        
        # Causal mask to prevent attending to future states
        mask = self._generate_square_subsequent_mask(seq_len, x.device)
        
        # Transformer forward pass. 
        # PyTorch TransformerEncoder with batch_first=True expects (batch, seq, feature)
        # The MHA will trigger the hooks and store attention weights.
        # Since PyTorch's default MHA in TransformerEncoderLayer doesn't return multi-head weights by default
        # (it averages them unless configured otherwise, which isn't exposed in TransformerEncoderLayer),
        # we still capture what is produced (batch, seq_len, seq_len) or (batch, nhead, seq_len, seq_len).
        output = self.transformer_encoder(x, mask=mask, is_causal=True)
        
        # We only need the representation of the last time step
        last_step_output = output[:, -1, :]
        
        return last_step_output, self.attention_weights

    def get_attention_weights(self) -> Dict[int, torch.Tensor]:
        """
        Returns the captured attention weights mapping layer_idx to attention tensor.
        """
        return self.attention_weights
