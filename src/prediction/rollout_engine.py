import torch
import numpy as np
from typing import Dict, List, Any

class RolloutEngine:
    """K-step forward simulation engine."""
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.model.eval()
        self.device = next(model.parameters()).device

    def rollout(self, current_states: torch.Tensor, k_steps: int = 10) -> dict:
        """
        K-step forward simulation.
        current_states: Tensor of shape [batch_size, context_length, state_dim]
        """
        states = current_states.to(self.device)
        batch_size = states.size(0)
        
        results: Dict[str, Any] = {
            'probabilities': [],
            'stages': [],
            'stage_names': [],
            'predicted_states': [],
            'confidence': []
        }
        
        from src.prediction.mitre_mapper import MitreMapper
        mapper = MitreMapper()
        
        with torch.no_grad():
            for step in range(k_steps):
                outputs = self.model(states)
                
                # Extract predictions
                if isinstance(outputs, dict):
                    next_state = outputs.get('next_state', states[:, -1, :])
                    inf_prob = outputs.get('infiltration_prob', torch.zeros(batch_size, 1, device=self.device)).squeeze(-1)
                    stage_logits = outputs.get('stage_logits', torch.zeros(batch_size, 6, device=self.device))
                else:
                    # Fallback for mocked model
                    next_state = states[:, -1, :]
                    inf_prob = torch.zeros(batch_size, device=self.device)
                    stage_logits = torch.zeros(batch_size, 6, device=self.device)
                
                # Update context window
                states = torch.cat([states[:, 1:, :], next_state.unsqueeze(1)], dim=1)
                
                stage_idx = torch.argmax(stage_logits, dim=-1)
                
                # Apply confidence decay
                decay = 0.95 ** step
                confidence = float(decay)
                inf_prob = inf_prob * decay
                
                # Append first element of batch for results
                results['probabilities'].append(inf_prob[0].item())
                idx = stage_idx[0].item()
                results['stages'].append(idx)
                results['stage_names'].append(mapper.get_stage_name(idx))
                results['predicted_states'].append(next_state[0])
                results['confidence'].append(confidence)
                
        results['predicted_states'] = torch.stack(results['predicted_states'], dim=0)
        return results

    def simulate_from_features(self, raw_features: np.ndarray, k_steps: int) -> dict:
        """Convenience method handling numpy arrays."""
        tensor_features = torch.tensor(raw_features, dtype=torch.float32).unsqueeze(0)
        return self.rollout(tensor_features, k_steps)
