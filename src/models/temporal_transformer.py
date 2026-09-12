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
        Wraps the MultiheadAttention forward on every encoder layer so that
        real per-head attention weights are captured during the forward pass.

        nn.TransformerEncoderLayer calls internal MHA with need_weights=False,
        which discards the weights. We override the MHA forward to request
        (batch, nhead, seq_len, seq_len) attention matrices and store them.
        """
        for i, layer in enumerate(self.transformer_encoder.layers):
            orig_forward = layer.self_attn.forward

            def make_wrapper(original, layer_idx):
                def wrapped(query, key, value, key_padding_mask=None, need_weights=True,
                            attn_mask=None, average_attn_weights=True, is_causal=False):
                    out, weights = original(
                        query, key, value,
                        key_padding_mask=key_padding_mask,
                        need_weights=True,
                        attn_mask=attn_mask,
                        average_attn_weights=False,
                        is_causal=is_causal,
                    )
                    if weights is not None:
                        self.attention_weights[layer_idx] = weights.detach()
                    else:
                        self.attention_weights.pop(layer_idx, None)
                    return out, weights
                return wrapped

            layer.self_attn.forward = make_wrapper(orig_forward, i)

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
