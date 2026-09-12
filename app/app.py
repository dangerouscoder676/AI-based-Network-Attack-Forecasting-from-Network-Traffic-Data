"""
AI Network Attack Forecasting — Main Streamlit Application
==========================================================

Premium dark-themed multi-page dashboard for network attack prediction
and analysis using World Models.

Launch with: streamlit run app/app.py
"""

import sys
import os

# Add project root and the app folder to path. The app folder must precede
# package resolution so `app.py` (the dashboard script) does not shadow the
# `app` package while Streamlit has the script directory on sys.path.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (_APP_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from theme import setup_page

import streamlit as st

# ─── Page Configuration + Dark Theme ────────────────────────────────────────
setup_page(title="NetSeer — AI Network Attack Forecaster")


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
