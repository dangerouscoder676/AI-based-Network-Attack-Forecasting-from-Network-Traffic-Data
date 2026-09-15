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
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return x


class TemporalEncoderLayer(nn.Module):
    """
    Pre-LN transformer encoder layer that is parameter-compatible with
    ``nn.TransformerEncoderLayer`` (norm_first=True, batch_first=True) but
    exposes the raw per-head attention weights on every forward pass.

    The built-in `nn.TransformerEncoderLayer` runs a fused C++ kernel in eval
    mode that bypasses the Python attention module, making weight extraction
    impossible. This layer avoids that fast path by calling the attention
    module explicitly, which is required for the explainability module.
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1, batch_first: bool = True):
        super(TemporalEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = True
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.GELU()

    def forward(self, src: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None,
                is_causal: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pre-LN encoder layer forward.

        Returns:
            x: transformed tensor of shape (batch, seq_len, d_model)
            attn_weights: attention of shape (batch, nhead, seq_len, seq_len)
        """
        x = src

        # Self-attention block (pre-LN)
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.self_attn(
            x_norm, x_norm, x_norm,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            attn_mask=src_mask,
            average_attn_weights=False,
            is_causal=is_causal,
        )
        x = x + self.dropout1(attn_out)

        # Feed-forward block (pre-LN)
        x_norm = self.norm2(x)
        ff_out = self.linear2(self.dropout(self.activation(self.linear1(x_norm))))
        x = x + self.dropout2(ff_out)

        return x, attn_weights


class TemporalTransformer(nn.Module):
    """
    Transformer for learning P(S_{t+1} | S_t) with causal masking and real
    per-head attention weight extraction for explainability.
    """
    def __init__(self, d_model: int = 128, nhead: int = 4, num_layers: int = 3,
                 dim_feedforward: int = 256, dropout: float = 0.15):
        super(TemporalTransformer, self).__init__()

        self.d_model = d_model
        self.pos_encoder = PositionalEncoding(d_model)
        self.attention_weights: Dict[int, torch.Tensor] = {}

        self.layers = nn.ModuleList([
            TemporalEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """
        Generates a causal float mask: 0 where attention is allowed, -inf elsewhere.
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
            attn_weights: Dictionary mapping layer index -> attention tensor
                          of shape (batch_size, nhead, seq_len, seq_len).
        """
        self.attention_weights.clear()

        seq_len = x.size(1)
        x = self.pos_encoder(x)

        mask = self._generate_square_subsequent_mask(seq_len, x.device)

        for i, layer in enumerate(self.layers):
            x, attn_weights = layer(x, src_mask=mask)
            self.attention_weights[i] = attn_weights.detach()

        # Representation of the last time step (the state we forecast from)
        last_step_output = x[:, -1, :]

        return last_step_output, self.attention_weights

    def get_attention_weights(self) -> Dict[int, torch.Tensor]:
        """
        Returns the captured attention weights mapping layer_idx to attention tensor.
        """
        return self.attention_weights