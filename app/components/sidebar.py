"""
Sidebar configuration component for NetSeer dashboard.
"""

import streamlit as st


def render_sidebar():
    """Render the sidebar with configuration options. Returns config dict."""

    st.sidebar.markdown("""
    <div style="text-align: center; padding: 10px 0 20px 0;">
        <span style="font-size: 2rem;">🛡️</span>
        <h2 style="margin: 4px 0; background: linear-gradient(135deg, #00ffaa, #00aaff);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            font-weight: 800; font-size: 1.6rem;">NetSeer</h2>
        <p style="color: rgba(255,255,255,0.4); font-size: 0.75rem; letter-spacing: 2px;">
            ATTACK FORECASTER v1.0
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("---")

    # ─── Data Input ──────────────────────────────────────────────────
    st.sidebar.markdown("### 📥 Data Input")
    data_source = st.sidebar.radio(
        "Data Source",
        ["Upload File", "Synthetic Demo", "Load from Directory"],
        index=1,
        help="Choose how to load network traffic data"
    )

    uploaded_file = None
    data_dir = None

    if data_source == "Upload File":
        uploaded_file = st.sidebar.file_uploader(
            "Upload CSV or PCAP",
            type=["csv", "pcap", "pcapng"],
            help="Upload CIC-IDS-2018 CSV or PCAP file"
        )
    elif data_source == "Load from Directory":
        data_dir = st.sidebar.text_input(
            "Data Directory",
            value="data/raw",
            help="Path to directory containing CSV files"
        )

    st.sidebar.markdown("---")

    # ─── Model Configuration ────────────────────────────────────────
    st.sidebar.markdown("### 🧠 Model Settings")
    model_type = st.sidebar.selectbox(
        "Model Architecture",
        ["Temporal Transformer", "LSTM Baseline"],
        index=0,
    )

    model_path = st.sidebar.text_input(
        "Model Weights Path",
        value="models/best_model.pt",
        help="Path to trained model weights"
    )

    st.sidebar.markdown("---")

    # ─── Prediction Settings ────────────────────────────────────────
    st.sidebar.markdown("### 🎯 Prediction")
    k_steps = st.sidebar.slider(
        "Forecast Horizon (K steps)",
        min_value=1, max_value=30, value=10,
        help="Number of future states to predict"
    )

    alert_threshold = st.sidebar.slider(
        "Alert Threshold",
        min_value=0.1, max_value=0.95, value=0.6, step=0.05,
        help="Minimum probability to trigger alerts"
    )

    st.sidebar.markdown("---")

    # ─── Explainability ─────────────────────────────────────────────
    st.sidebar.markdown("### 🔍 Explainability")
    xai_method = st.sidebar.selectbox(
        "XAI Method",
        ["Both (Attention + SHAP)", "Attention Only", "SHAP Only"],
        index=0,
    )

    top_k_features = st.sidebar.slider(
        "Top Features to Show",
        min_value=5, max_value=30, value=10,
    )

    st.sidebar.markdown("---")

    # ─── System Status ──────────────────────────────────────────────
    st.sidebar.markdown("### ⚡ System Status")

    import torch
    device = "CUDA" if torch.cuda.is_available() else "CPU"
    device_color = "#00ffaa" if device == "CUDA" else "#ffcc00"

    st.sidebar.markdown(f"""
    <div style="background: rgba(0,255,170,0.05); border: 1px solid rgba(0,255,170,0.15);
         border-radius: 8px; padding: 12px; margin: 8px 0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
            <span style="color: rgba(255,255,255,0.6); font-size: 0.8rem;">Device</span>
            <span style="color: {device_color}; font-weight: 600; font-size: 0.8rem;">{device}</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
            <span style="color: rgba(255,255,255,0.6); font-size: 0.8rem;">PyTorch</span>
            <span style="color: #00aaff; font-weight: 600; font-size: 0.8rem;">{torch.__version__}</span>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span style="color: rgba(255,255,255,0.6); font-size: 0.8rem;">Mode</span>
            <span style="color: #00ffaa; font-weight: 600; font-size: 0.8rem;">Offline</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    return {
        "data_source": data_source,
        "uploaded_file": uploaded_file,
        "data_dir": data_dir,
        "model_type": "transformer" if "Transformer" in model_type else "lstm",
        "model_path": model_path,
        "k_steps": k_steps,
        "alert_threshold": alert_threshold,
        "xai_method": xai_method.lower().split(" ")[0],
        "top_k_features": top_k_features,
    }
