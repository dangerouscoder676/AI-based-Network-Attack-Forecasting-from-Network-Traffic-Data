import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple
import warnings
import shap

class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        outputs = self.model(x)
        if isinstance(outputs, dict) and 'infiltration_prob' in outputs:
            return outputs['infiltration_prob']
        # Ensure 1D output
        return outputs.mean(dim=-1, keepdim=True) if hasattr(outputs, 'mean') else torch.zeros(x.size(0), 1).to(x.device)

class ShapExplainer:
    """SHAP feature attribution."""
    def __init__(self, model: nn.Module, background_data: np.ndarray):
        self.wrapper = ModelWrapper(model)
        self.wrapper.eval()
        self.device = next(model.parameters()).device
        self.background = torch.tensor(background_data, dtype=torch.float32).to(self.device)
        
        # Disable cudnn for SHAP compatibility
        torch.backends.cudnn.enabled = False
        
        try:
            self.explainer = shap.GradientExplainer(self.wrapper, self.background)
            self.method = "gradient"
        except Exception:
            def predict_func(x):
                x_t = torch.tensor(x, dtype=torch.float32).to(self.device)
                with torch.no_grad():
                    return self.wrapper(x_t).cpu().numpy()
            self.explainer = shap.KernelExplainer(predict_func, background_data[:10])
            self.method = "kernel"

    def explain(self, input_data: np.ndarray, target_class: int = None) -> dict:
        in_tensor = torch.tensor(input_data, dtype=torch.float32, requires_grad=True).to(self.device)
        
        try:
            if self.method == "gradient":
                shap_values = self.explainer.shap_values(in_tensor)
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]
            else:
                shap_values = self.explainer.shap_values(input_data)
        except Exception as e:
            shap_values = np.zeros(input_data.shape)

        if isinstance(shap_values, torch.Tensor):
            shap_values = shap_values.cpu().detach().numpy()

        abs_vals = np.abs(shap_values)
        if abs_vals.ndim == 4:
            # [samples, time, features, outputs]
            feat_imp = abs_vals.mean(axis=(0, 3)).mean(axis=0)
        elif abs_vals.ndim == 3:
            # [samples, features, outputs]
            feat_imp = abs_vals.mean(axis=(0, 2))
        elif abs_vals.ndim == 2:
            feat_imp = abs_vals.mean(axis=0)
        else:
            feat_imp = np.zeros_like(np.asarray(shap_values, dtype=np.float64))

        return {
            'shap_values': shap_values,
            'feature_importance': feat_imp,
            'top_features': []
        }

    def get_top_k_features(self, shap_values: np.ndarray, feature_names: list, k: int = 10) -> List[Tuple[str, float]]:
        abs_vals = np.abs(shap_values)
        if abs_vals.ndim == 4:
            imp = abs_vals.mean(axis=(0, 3)).mean(axis=0)
        elif abs_vals.ndim == 3:
            imp = abs_vals.mean(axis=(0, 2))
        else:
            imp = abs_vals.mean(axis=0)

        indices = np.argsort(imp)[-k:][::-1]
        return [(feature_names[i], float(imp[i])) for i in indices if i < len(feature_names)]
