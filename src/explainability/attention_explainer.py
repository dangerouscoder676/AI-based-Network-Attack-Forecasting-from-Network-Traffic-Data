import torch
import numpy as np

class AttentionExplainer:
    """Transformer attention extraction.

    Pulls the real per-head attention matrices captured by the
    TemporalTransformer during its forward pass and aggregates them
    into temporal + per-layer views for explainability.
    """
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.model.eval()

    def _get_attention_dict(self):
        dynamics = getattr(self.model, 'dynamics_model', None)
        if dynamics is None:
            return None
        return getattr(dynamics, 'attention_weights', None)

    def explain(self, input_sequence: torch.Tensor) -> dict:
        """
        Run the model on the input and extract attention from every layer.

        input_sequence: [batch, seq_len, features]

        Returns dict with:
            - 'temporal_attention': [seq_len] normalized attention mass per time step
            - 'layer_attention': {layer_idx: [batch, heads, seq_len, seq_len] or [heads, seq, seq]}
            - 'feature_importance': placeholder (feature-level attribution comes from SHAP)
        """
        device = next(self.model.parameters()).device
        inputs = input_sequence.to(device)

        batch, seq_len, features = inputs.shape

        with torch.no_grad():
            outputs = self.model(inputs)

        attention_weights = self._get_attention_dict()

        if attention_weights is None or len(attention_weights) == 0:
            # Model has no extractable attention (e.g. LSTM) -> uniform fallback.
            fallback = torch.ones(batch, 1, seq_len, seq_len, device=device) / seq_len
            attention_weights = {0: fallback}

        layer_attention = {}
        stacked = []
        for idx, w in sorted(attention_weights.items()):
            w_arr = w.cpu().numpy()
            # w_arr may be [batch, heads, seq, seq] or [heads, seq, seq]
            if w_arr.ndim == 4:
                layer_attention[idx] = w_arr
                stacked.append(w_arr)
            elif w_arr.ndim == 3:
                layer_attention[idx] = w_arr[None, ...]
                stacked.append(w_arr[None, ...])
            else:
                continue

        if not stacked:
            return {
                'temporal_attention': np.ones(seq_len) / seq_len,
                'layer_attention': {},
                'feature_importance': np.zeros(features),
            }

        all_weights = np.concatenate(stacked, axis=0)  # [layers, batch, heads, seq, seq]

        # Temporal attention: mean over layers, heads, and batch -> column sum
        mean_attn = all_weights.mean(axis=(0, 2))      # [batch, seq, seq]
        mean_attn = mean_attn[0]                       # [seq, seq]
        # Normalize each row and sum along keys -> attention mass per position
        row_sums = mean_attn.sum(axis=1, keepdims=True) + 1e-8
        temporal = mean_attn.sum(axis=1) / row_sums[:, 0]
        temporal = temporal / (temporal.sum() + 1e-8)

        return {
            'temporal_attention': temporal.tolist(),
            'layer_attention': layer_attention,
            'feature_importance': np.zeros(features),
        }

    def get_attention_heatmap(self, attention_weights: dict, layer: int = -1) -> np.ndarray:
        """Return a [seq_len, seq_len] attention heatmap averaged over batch+heads."""
        if layer == -1:
            layer = max(attention_weights.keys())
        attn = attention_weights[layer]
        attn = attn[0] if attn.ndim == 4 else attn          # first batch element
        attn = attn.mean(axis=0) if attn.ndim == 3 else attn  # average over heads
        return attn