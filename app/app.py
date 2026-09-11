"""
AI Network Attack Forecasting — Main Streamlit Application
==========================================================

Premium dark-themed multi-page dashboard for network attack prediction
and analysis using World Models.

Launch with: streamlit run app/app.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

# ─── Page Configuration ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="NetSeer — AI Network Attack Forecaster",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "# NetSeer v1.0\nAI-Based Network Attack Forecasting using World Models"
    }
)

# ─── Custom CSS — Premium Dark Theme ────────────────────────────────────────
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Main background */
    .stApp > header {
        background-color: transparent;
    }

    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #0d1525 40%, #0a1628 100%);
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1525 0%, #111d33 100%);
        border-right: 1px solid rgba(0, 255, 170, 0.1);
    }

    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #00ffaa;
    }

    /* Cards / Containers */
    [data-testid="stVerticalBlock"] > div > div[data-testid="stExpander"] {
        background: rgba(13, 21, 37, 0.8);
        border: 1px solid rgba(0, 255, 170, 0.15);
        border-radius: 12px;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(0, 255, 170, 0.05) 0%, rgba(0, 170, 255, 0.05) 100%);
        border: 1px solid rgba(0, 255, 170, 0.15);
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }

    div[data-testid="stMetric"] label {
        color: rgba(255, 255, 255, 0.6) !important;
        font-size: 0.85rem !important;
        font-weight: 500;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }

    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #00ffaa !important;
        font-weight: 700;
        font-size: 1.8rem !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #00ffaa 0%, #00aaff 100%);
        color: #0a0e1a;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 255, 170, 0.3);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(0, 255, 170, 0.5);
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        border: 2px dashed rgba(0, 255, 170, 0.3);
        border-radius: 12px;
        padding: 20px;
        background: rgba(0, 255, 170, 0.02);
        transition: all 0.3s ease;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: rgba(0, 255, 170, 0.6);
        background: rgba(0, 255, 170, 0.05);
    }

    /* Headers */
    h1 {
        background: linear-gradient(135deg, #00ffaa, #00aaff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        letter-spacing: -0.5px;
    }

    h2, h3 {
        color: #e0e8f0;
        font-weight: 600;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(0, 255, 170, 0.05);
        border: 1px solid rgba(0, 255, 170, 0.15);
        border-radius: 8px;
        color: rgba(255, 255, 255, 0.7);
        font-weight: 500;
        padding: 8px 20px;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(0, 255, 170, 0.15) 0%, rgba(0, 170, 255, 0.15) 100%);
        border-color: #00ffaa;
        color: #00ffaa !important;
    }

    /* Selectbox & Slider */
    [data-testid="stSelectbox"] > div > div {
        background: rgba(13, 21, 37, 0.8);
        border: 1px solid rgba(0, 255, 170, 0.2);
        border-radius: 8px;
    }

    /* Alert boxes */
    .alert-critical {
        background: linear-gradient(135deg, rgba(255, 51, 102, 0.15) 0%, rgba(255, 51, 102, 0.05) 100%);
        border: 1px solid rgba(255, 51, 102, 0.4);
        border-radius: 10px;
        padding: 16px 20px;
        margin: 8px 0;
    }

    .alert-high {
        background: linear-gradient(135deg, rgba(255, 153, 0, 0.15) 0%, rgba(255, 153, 0, 0.05) 100%);
        border: 1px solid rgba(255, 153, 0, 0.4);
        border-radius: 10px;
        padding: 16px 20px;
        margin: 8px 0;
    }

    .alert-medium {
        background: linear-gradient(135deg, rgba(255, 204, 0, 0.15) 0%, rgba(255, 204, 0, 0.05) 100%);
        border: 1px solid rgba(255, 204, 0, 0.4);
        border-radius: 10px;
        padding: 16px 20px;
        margin: 8px 0;
    }

    .alert-low {
        background: linear-gradient(135deg, rgba(0, 255, 170, 0.1) 0%, rgba(0, 255, 170, 0.03) 100%);
        border: 1px solid rgba(0, 255, 170, 0.3);
        border-radius: 10px;
        padding: 16px 20px;
        margin: 8px 0;
    }

    /* Status indicator */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    .status-safe { background: rgba(0, 255, 170, 0.15); color: #00ffaa; border: 1px solid rgba(0, 255, 170, 0.3); }
    .status-warning { background: rgba(255, 204, 0, 0.15); color: #ffcc00; border: 1px solid rgba(255, 204, 0, 0.3); }
    .status-danger { background: rgba(255, 51, 102, 0.15); color: #ff3366; border: 1px solid rgba(255, 51, 102, 0.3); }

    /* Kill chain progress */
    .kill-chain-container {
        display: flex;
        gap: 4px;
        margin: 12px 0;
    }

    .kill-chain-step {
        flex: 1;
        padding: 10px;
        text-align: center;
        border-radius: 8px;
        font-size: 0.75rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .kill-chain-active {
        background: linear-gradient(135deg, #ff3366, #ff6633);
        color: white;
        box-shadow: 0 4px 15px rgba(255, 51, 102, 0.4);
    }

    .kill-chain-inactive {
        background: rgba(255, 255, 255, 0.05);
        color: rgba(255, 255, 255, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .kill-chain-passed {
        background: rgba(255, 51, 102, 0.2);
        color: rgba(255, 51, 102, 0.8);
        border: 1px solid rgba(255, 51, 102, 0.3);
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0a0e1a; }
    ::-webkit-scrollbar-thumb { background: rgba(0, 255, 170, 0.3); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(0, 255, 170, 0.5); }

    /* Animation for threat pulse */
    @keyframes threatPulse {
        0% { box-shadow: 0 0 5px rgba(255, 51, 102, 0.3); }
        50% { box-shadow: 0 0 20px rgba(255, 51, 102, 0.6); }
        100% { box-shadow: 0 0 5px rgba(255, 51, 102, 0.3); }
    }

    .threat-active {
        animation: threatPulse 2s infinite;
    }

    /* Glassmorphism card */
    .glass-card {
        background: rgba(13, 21, 37, 0.6);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 255, 170, 0.1);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }

    /* Data table */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }

    /* Divider */
    hr {
        border-color: rgba(0, 255, 170, 0.1);
    }
</style>
""", unsafe_allow_html=True)


# ─── Landing Page ────────────────────────────────────────────────────────────
def main():
    """Main landing page."""

    # Hero Section
    st.markdown("""
    <div style="text-align: center; padding: 40px 0 20px 0;">
        <h1 style="font-size: 3.2rem; margin-bottom: 8px;">🛡️ NetSeer</h1>
        <p style="color: rgba(255,255,255,0.5); font-size: 1.1rem; font-weight: 300; letter-spacing: 1px;">
            AI-BASED NETWORK ATTACK FORECASTING USING WORLD MODELS
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Feature Cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="glass-card" style="text-align: center; min-height: 220px;">
            <div style="font-size: 2.5rem; margin-bottom: 12px;">🧠</div>
            <h3 style="color: #00ffaa; margin-bottom: 8px;">World Model Engine</h3>
            <p style="color: rgba(255,255,255,0.6); font-size: 0.9rem;">
                Temporal Transformer learns network state dynamics P(S<sub>t+1</sub> | S<sub>t</sub>) — 
                predicts future states, not just classifies current ones.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="glass-card" style="text-align: center; min-height: 220px;">
            <div style="font-size: 2.5rem; margin-bottom: 12px;">🎯</div>
            <h3 style="color: #00aaff; margin-bottom: 8px;">K-Step Forecasting</h3>
            <p style="color: rgba(255,255,255,0.6); font-size: 0.9rem;">
                Forward simulation rolls out K future states to predict infiltration 
                probability and MITRE ATT&CK stage before compromise.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="glass-card" style="text-align: center; min-height: 220px;">
            <div style="font-size: 2.5rem; margin-bottom: 12px;">🔍</div>
            <h3 style="color: #ff9900; margin-bottom: 8px;">Explainable AI</h3>
            <p style="color: rgba(255,255,255,0.6); font-size: 0.9rem;">
                SHAP values and attention mechanisms reveal exactly which traffic 
                features drive each prediction — no black boxes.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Quick Start
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color: #00ffaa;">🚀 Quick Start</h3>
            <ol style="color: rgba(255,255,255,0.7); line-height: 2;">
                <li>Navigate to <strong style="color: #00ffaa;">📊 Dashboard</strong> in the sidebar</li>
                <li>Upload a CSV or PCAP file</li>
                <li>View real-time infiltration forecasts</li>
                <li>Explore explainability in <strong style="color: #00aaff;">🔍 Analysis</strong></li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color: #00aaff;">📋 System Capabilities</h3>
            <ul style="color: rgba(255,255,255,0.7); line-height: 2;">
                <li>CIC-IDS-2018 / CTU-13 dataset support</li>
                <li>PCAP file parsing via Scapy</li>
                <li>MITRE ATT&CK kill chain mapping</li>
                <li>Fully offline — no cloud dependencies</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Architecture Preview
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card">
        <h3 style="color: #e0e8f0; text-align: center; margin-bottom: 16px;">System Architecture</h3>
        <div style="display: flex; justify-content: center; gap: 12px; flex-wrap: wrap; padding: 10px;">
            <div style="background: rgba(0,255,170,0.1); border: 1px solid rgba(0,255,170,0.3); border-radius: 8px; padding: 10px 18px; text-align: center;">
                <div style="color: #00ffaa; font-weight: 600;">📥 Ingestion</div>
                <div style="color: rgba(255,255,255,0.5); font-size: 0.75rem;">CSV / PCAP</div>
            </div>
            <div style="color: rgba(255,255,255,0.3); display: flex; align-items: center; font-size: 1.5rem;">→</div>
            <div style="background: rgba(0,170,255,0.1); border: 1px solid rgba(0,170,255,0.3); border-radius: 8px; padding: 10px 18px; text-align: center;">
                <div style="color: #00aaff; font-weight: 600;">⚙️ Features</div>
                <div style="color: rgba(255,255,255,0.5); font-size: 0.75rem;">76-dim State</div>
            </div>
            <div style="color: rgba(255,255,255,0.3); display: flex; align-items: center; font-size: 1.5rem;">→</div>
            <div style="background: rgba(170,0,255,0.1); border: 1px solid rgba(170,0,255,0.3); border-radius: 8px; padding: 10px 18px; text-align: center;">
                <div style="color: #aa66ff; font-weight: 600;">🧠 World Model</div>
                <div style="color: rgba(255,255,255,0.5); font-size: 0.75rem;">Transformer</div>
            </div>
            <div style="color: rgba(255,255,255,0.3); display: flex; align-items: center; font-size: 1.5rem;">→</div>
            <div style="background: rgba(255,153,0,0.1); border: 1px solid rgba(255,153,0,0.3); border-radius: 8px; padding: 10px 18px; text-align: center;">
                <div style="color: #ff9900; font-weight: 600;">🎯 Prediction</div>
                <div style="color: rgba(255,255,255,0.5); font-size: 0.75rem;">K-Step Rollout</div>
            </div>
            <div style="color: rgba(255,255,255,0.3); display: flex; align-items: center; font-size: 1.5rem;">→</div>
            <div style="background: rgba(255,51,102,0.1); border: 1px solid rgba(255,51,102,0.3); border-radius: 8px; padding: 10px 18px; text-align: center;">
                <div style="color: #ff3366; font-weight: 600;">🔍 Explain</div>
                <div style="color: rgba(255,255,255,0.5); font-size: 0.75rem;">SHAP + Attention</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Footer
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; padding: 20px; color: rgba(255,255,255,0.3); font-size: 0.8rem;">
        NetSeer v1.0 — AI-Based Network Attack Forecasting System<br/>
        Powered by PyTorch • Streamlit • MITRE ATT&CK Framework
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
