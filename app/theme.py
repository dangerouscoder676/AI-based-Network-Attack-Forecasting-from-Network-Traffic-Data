"""
NetSeer shared dark theme (CSS + page config).
"""

import streamlit as st

DARK_THEME_CSS = """
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

    /* One-click console: stat tiles, device cards, door chips */
    .stat-tile {
        background: rgba(13, 21, 37, 0.6);
        border: 1px solid rgba(0, 255, 170, 0.12);
        border-radius: 14px;
        padding: 16px 18px;
        text-align: center;
        height: 100%;
    }
    .stat-tile .stat-icon { font-size: 1.7rem; }
    .stat-tile .stat-value { font-size: 1.7rem; font-weight: 800; margin: 2px 0; }
    .stat-tile .stat-label { font-size: 0.78rem; color: rgba(255, 255, 255, 0.5); letter-spacing: 0.5px; }

    .device-card {
        background: rgba(13, 21, 37, 0.55);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 14px 16px;
        height: 100%;
        border-left: 3px solid #3fb950;
    }
    .device-card.warn { border-left-color: #d29922; }
    .device-card.danger { border-left-color: #f85149; }

    .door-chip {
        display: inline-block;
        margin: 3px 3px 0 0;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.74rem;
        background: rgba(255, 153, 0, 0.12);
        border: 1px solid rgba(255, 153, 0, 0.35);
        color: #ffb366;
        white-space: nowrap;
    }
    .door-chip.locked {
        background: rgba(0, 255, 170, 0.08);
        border-color: rgba(0, 255, 170, 0.25);
        color: #00ffaa;
    }

    .howto-card {
        background: rgba(13, 21, 37, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 12px;
        padding: 14px 16px;
        height: 100%;
    }
</style>
"""


def setup_page(title="NetSeer", subtitle=None):
    """Configure the Streamlit page + inject the dark theme."""
    st.set_page_config(
        page_title=title,
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Get Help": None,
            "Report a bug": None,
            "About": "# NetSeer v1.0\nAI-Based Network Attack Forecasting using World Models"
        }
    )
    st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)
    if subtitle:
        st.markdown(
            f'<div style="text-align:center; padding:18px 0 4px 0;">'
            f'<h1 style="font-size:2.2rem; margin-bottom:4px;">{title}</h1>'
            f'<p style="color:rgba(255,255,255,0.5); font-size:0.95rem; font-weight:300; '
            f'letter-spacing:1px;">{subtitle}</p></div>',
            unsafe_allow_html=True,
        )