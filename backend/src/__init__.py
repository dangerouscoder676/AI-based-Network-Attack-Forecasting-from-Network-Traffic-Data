"""
AI-Based Network Attack Forecasting from Network Traffic Data
=============================================================

A World Model architecture that learns network state transition dynamics
P(S_t+1 | S_t) from traffic telemetry and predicts attacker progression
before compromise is completed.

Components:
    - data: Feature extraction pipeline (CSV, PCAP, synthetic)
    - models: World Model architectures (Transformer, LSTM, baselines)
    - prediction: K-step forward simulation and MITRE ATT&CK mapping
    - explainability: SHAP and attention-based interpretability
    - utils: Configuration, metrics, and visualization utilities
"""

__version__ = "1.0.0"
__author__ = "Network Attack Forecasting Team"
