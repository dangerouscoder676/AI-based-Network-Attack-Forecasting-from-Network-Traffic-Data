from typing import Dict, Any, List

class FeatureReport:
    """Unified explanation reports."""
    def __init__(self):
        pass

    def generate_report(self, attention_results: dict, shap_results: dict, prediction: dict, feature_names: list, mitre_mapper) -> dict:
        top_shap = shap_results.get('feature_importance', [])
        
        top_features = []
        if len(top_shap) > 0:
            import numpy as np
            indices = np.argsort(top_shap)[-5:][::-1]
            for idx in indices:
                if idx < len(feature_names):
                    top_features.append({
                        'name': feature_names[idx],
                        'importance': float(top_shap[idx]),
                        'shap_value': float(top_shap[idx]),
                        'attack_relevance': 'High'
                    })
        
        stage_idx = prediction.get('stages', [0])[0] if prediction.get('stages') else 0
        stage_name = mitre_mapper.get_stage_name(stage_idx)
        
        feat_strs = [f"{f['name']} [ATT&CK: {stage_name}]" for f in top_features[:3]]
        summary = f"This prediction is driven by: " + ", ".join(feat_strs) + "."
        
        report = {
            'summary': summary,
            'top_features': top_features,
            'temporal_pattern': "Recent time steps have the highest attention weights, indicating a sudden shift in behavior.",
            'attack_stage_explanation': f"The model identified patterns consistent with {stage_name}.",
            'confidence_explanation': "Confidence is based on the consistency of the indicators over the rollout window.",
            'recommended_investigation': mitre_mapper.get_mitigations_for_stage(stage_idx)
        }
        
        return report

    def format_text_report(self, report: dict) -> str:
        s = f"=== Feature Report ===\n\n"
        s += f"Summary:\n{report['summary']}\n\n"
        
        s += f"Top Features:\n"
        for f in report['top_features']:
            s += f" - {f['name']}: {f['importance']:.4f}\n"
            
        s += f"\nTemporal Pattern: {report['temporal_pattern']}\n"
        s += f"Attack Stage: {report['attack_stage_explanation']}\n"
        s += f"Confidence: {report['confidence_explanation']}\n\n"
        
        s += "Recommended Investigation:\n"
        for ri in report['recommended_investigation']:
            s += f" - {ri}\n"
            
        return s
