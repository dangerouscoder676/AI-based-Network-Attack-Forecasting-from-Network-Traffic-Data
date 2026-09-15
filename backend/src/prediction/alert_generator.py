from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any
import numpy as np

@dataclass
class Alert:
    timestamp: str
    severity: str
    probability: float
    stage_name: str
    stage_idx: int
    contributing_features: List[Tuple[str, float]]
    recommended_actions: List[str]
    flow_details: Dict[str, Any]

class AlertGenerator:
    """Alert generation."""
    def __init__(self):
        self.thresholds = {
            'LOW': 0.3,
            'MEDIUM': 0.6,
            'HIGH': 0.8,
            'CRITICAL': 0.95
        }
        
    def generate_alerts(self, rollout_results: dict, feature_names: list, feature_values: np.ndarray, timestamp: str = None) -> List[Alert]:
        from src.prediction.mitre_mapper import MitreMapper
        mapper = MitreMapper()
        
        alerts = []
        
        for i in range(len(rollout_results['probabilities'])):
            prob = rollout_results['probabilities'][i]
            if prob > self.thresholds['LOW']:
                severity = 'LOW'
                if prob > self.thresholds['CRITICAL']:
                    severity = 'CRITICAL'
                elif prob > self.thresholds['HIGH']:
                    severity = 'HIGH'
                elif prob > self.thresholds['MEDIUM']:
                    severity = 'MEDIUM'
                    
                stage_idx = rollout_results['stages'][i]
                
                # Mock feature contribution
                contrib = []
                for j in range(min(5, len(feature_names))):
                    contrib.append((feature_names[j], float(feature_values[-1, j] if len(feature_values.shape) > 1 else feature_values[j])))
                
                ts = timestamp if timestamp else f"t+{i}"
                
                alert = Alert(
                    timestamp=ts,
                    severity=severity,
                    probability=prob,
                    stage_name=mapper.get_stage_name(stage_idx),
                    stage_idx=stage_idx,
                    contributing_features=contrib,
                    recommended_actions=mapper.get_mitigations_for_stage(stage_idx),
                    flow_details={}
                )
                alerts.append(alert)
                
        return alerts

    def format_alert(self, alert: Alert) -> str:
        s = f"[{alert.severity}] Alert at {alert.timestamp}: {alert.stage_name} detected with probability {alert.probability:.2f}.\n"
        s += "Key indicators:\n"
        for f, v in alert.contributing_features:
            s += f"  - {f}: {v}\n"
        s += "Recommended Actions:\n"
        for a in alert.recommended_actions:
            s += f"  - {a}\n"
        return s

    def to_dict(self, alert: Alert) -> dict:
        return asdict(alert)

    def get_severity_color(self, severity: str) -> str:
        colors = {
            'LOW': '#FFFF00',      # Yellow
            'MEDIUM': '#FFA500',   # Orange
            'HIGH': '#FF0000',     # Red
            'CRITICAL': '#8B0000'  # Dark Red
        }
        return colors.get(severity, '#FFFFFF')
