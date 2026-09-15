import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from .state_encoder import StateEncoder
from .temporal_transformer import TemporalTransformer
from .lstm_baseline import LSTMDynamics
from .infiltration_head import InfiltrationHead

class WorldModel(nn.Module):
    """
    Main World Model architecture that combines State Encoder, Dynamics Model, 
    Infiltration Head, and State Decoder.
    """
    def __init__(
        self, 
        input_dim: int = 76,
        latent_dim: int = 128,
        hidden_dim: int = 192,
        num_stages: int = 6,
        dynamics_type: str = "transformer",
        dropout: float = 0.15,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        loss_weights: Dict[str, float] = None
    ):
        super(WorldModel, self).__init__()
        
        if loss_weights is None:
            loss_weights = {"transition": 1.0, "infiltration": 2.0, "stage": 1.5}
        self.loss_weights = loss_weights
        self.input_dim = input_dim
        
        # 1. State Encoder (includes decoder)
        self.state_encoder = StateEncoder(
            input_dim=input_dim, 
            hidden=hidden_dim, 
            latent_dim=latent_dim, 
            dropout=dropout
        )
        
        # 2. Dynamics Model
        if dynamics_type.lower() == "transformer":
            self.dynamics_model = TemporalTransformer(
                d_model=latent_dim,
                nhead=nhead,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            )
        elif dynamics_type.lower() == "lstm":
            self.dynamics_model = LSTMDynamics(
                d_model=latent_dim,
                hidden_dim=latent_dim,
                num_layers=num_layers,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unknown dynamics_type: {dynamics_type}")
            
        # 3. Infiltration Head
        self.infiltration_head = InfiltrationHead(
            latent_dim=latent_dim,
            num_stages=num_stages
        )
        
        # Loss functions
        self.transition_criterion = nn.SmoothL1Loss()
        self.infiltration_criterion = nn.BCELoss()
        self.stage_criterion = nn.CrossEntropyLoss()
        
    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Forward pass for the World Model.
        
        Args:
            x: Raw feature sequence of shape (batch, seq_len, input_dim)
            
        Returns:
            dict with keys:
            - 'next_state_pred': Predicted next state features (batch, input_dim)
            - 'infiltration_prob': Infiltration probability (batch, 1)
            - 'stage_logits': Stage classification logits (batch, num_stages)
            - 'attention_weights': Attention weights from dynamics model
            - 'latent_states': Encoded latent states (batch, seq_len, latent_dim)
        """
        batch_size, seq_len, _ = x.size()
        
        # 1. Encode raw features -> latent states
        # Flatten batch and seq_len for MLP processing
        x_flat = x.view(-1, self.input_dim)
        z_flat = self.state_encoder(x_flat)
        latent_states = z_flat.view(batch_size, seq_len, -1)
        
        # 2. Dynamics model -> context vector + attention
        context, attention_weights = self.dynamics_model(latent_states)
        
        # 3. Predict next state
        next_state_pred = self.state_encoder.decode(context)
        
        # 4. Predict infiltration
        infiltration_prob, stage_logits = self.infiltration_head(context)
        
        return {
            'next_state_pred': next_state_pred,
            'infiltration_prob': infiltration_prob,
            'stage_logits': stage_logits,
            'attention_weights': attention_weights,
            'latent_states': latent_states
        }
        
    def compute_loss(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Computes the weighted sum of losses.
        
        Args:
            predictions: Output dict from forward pass.
            targets: Dict containing 'next_state' (batch, input_dim), 'infiltration' (batch, 1), 'stage' (batch,)
            
        Returns:
            Dict containing 'transition_loss', 'infiltration_loss', 'stage_loss', and 'total_loss'.
        """
        # Transition loss (Smooth L1)
        transition_loss = self.transition_criterion(
            predictions['next_state_pred'], 
            targets['next_state']
        )
        
        # Support flexible target key names
        infil_target = targets.get('infiltration', targets.get('infiltration_label'))
        stage_target = targets.get('stage', targets.get('stage_label'))

        # Infiltration loss (BCE)
        infil_tensor = infil_target.float().view_as(predictions['infiltration_prob'])
        infiltration_loss = self.infiltration_criterion(
            predictions['infiltration_prob'], 
            infil_tensor
        )
        
        # Stage loss (CrossEntropy)
        stage_loss = self.stage_criterion(
            predictions['stage_logits'], 
            stage_target.long()
        )
        
        total_loss = (
            self.loss_weights['transition'] * transition_loss +
            self.loss_weights['infiltration'] * infiltration_loss +
            self.loss_weights['stage'] * stage_loss
        )
        
        return {
            'transition_loss': transition_loss,
            'infiltration_loss': infiltration_loss,
            'stage_loss': stage_loss,
            'total_loss': total_loss
        }
        
    @classmethod
    def from_config(cls, config):
        """
        Instantiate WorldModel from a configuration dictionary or Config object.
        """
        if hasattr(config, 'as_dict'):
            config = config.as_dict()
        input_dim = config.get('features', {}).get('num_state_features', 76)
        latent_dim = config.get('model', {}).get('latent_dim', 128)
        num_stages = config.get('model', {}).get('num_attack_stages', 6)
        dynamics_type = config.get('model', {}).get('type', 'transformer')
        dropout = config.get('model', {}).get('dropout', 0.15)
        nhead = config.get('model', {}).get('num_heads', 4)
        num_layers = config.get('model', {}).get('num_layers', 3)
        dim_feedforward = config.get('model', {}).get('ffn_dim', 256)
        
        # Default loss weights
        loss_weights = {
            'transition': 1.0,
            'infiltration': 2.0,
            'stage': 1.5
        }
        
        # Override with config if present
        config_loss_weights = config.get('training', {}).get('loss_weights', {})
        if config_loss_weights:
            loss_weights.update(config_loss_weights)
            
        return cls(
            input_dim=input_dim,
            latent_dim=latent_dim,
            num_stages=num_stages,
            dynamics_type=dynamics_type,
            dropout=dropout,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            loss_weights=loss_weights
        )
