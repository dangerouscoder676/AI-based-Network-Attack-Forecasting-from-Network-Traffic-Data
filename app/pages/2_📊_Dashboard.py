"""
NetSeer — Dashboard
===================
Live attack forecasting view: real-time monitoring of infiltration
probability, predicted attack stages, and generated alerts.
"""

import sys
import os

_PAGES_DIR = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.dirname(_PAGES_DIR)
_PROJECT_ROOT = os.path.dirname(_APP_DIR)
for _p in (_APP_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import streamlit as st

from theme import setup_page
from components.sidebar import render_sidebar
from session import get_config, get_data, load_model, run_rollout, run_attention, generate_alerts
from components.charts import (
    create_gauge_chart,
    create_infiltration_timeline,
    create_attack_stage_timeline,
    create_network_state_heatmap,
    create_attention_heatmap,
    create_kill_chain_html,
    SEVERITY_COLORS,
    STAGE_NAMES,
)

setup_page("NetSeer", "Real-Time Attack Forecasting Dashboard")

st.markdown("<div style='text-align:center;'>")
st.markdown("<h2>Dashboard</h2>", unsafe_allow_html=True)
st.markdown("")
st.markdown("</div>", unsafe_allow_html=True)

settings = render_sidebar()

config = get_config()
context_length = int(config["data"]["context_length"])

if settings["data_source"] == "Upload File" and settings["uploaded_file"] is None:
    st.info("👈 Upload a CIC-IDS-2018 CSV, a PCAP, or a NetSeer state file in the sidebar to begin.")
    st.stop()


# ─── Data ────────────────────────────────────────────────────────────────────
data = get_data(settings, context_length)
if data is None:
    st.stop()

contexts, seq_stages, feature_names, states, timeline_stages = data
n_contexts = len(contexts)

if n_contexts == 0:
    st.error("No context windows available in the selected data.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.markdown("### 🗂️ Monitor Options")
default_idx = min(int(n_contexts * 0.75), n_contexts - 1)
context_index = st.sidebar.slider(
    "Context Window Index",
    min_value=0,
    max_value=n_contexts - 1,
    value=default_idx,
    help="Which recent window of network activity to analyze",
)

# ─── Model ───────────────────────────────────────────────────────────────────
model = load_model(settings["model_type"], settings["model_path"])
if model is None:
    st.error("No trained model weights found. Run `python scripts/train.py` "
             "before using the dashboard, or select a valid weights path in the sidebar.")
    st.stop()

# ─── Inference ───────────────────────────────────────────────────────────────
context = np.asarray(contexts[context_index])
true_stage = int(seq_stages[context_index])
true_stage_name = STAGE_NAMES.get(true_stage, "Unknown")

with st.spinner("Simulating future network states..."):
    rollout = run_rollout(model, context, settings["k_steps"])
    attention = run_attention(model, context)

if rollout is None:
    st.stop()

probs = np.asarray(rollout["probabilities"])
pred_stages = np.asarray(rollout["stages"])
confidences = np.asarray(rollout["confidence"])

best_step = int(np.argmax(probs))
current_threat = float(probs.max())
predicted_stage = int(pred_stages[best_step])
predicted_stage_name = STAGE_NAMES.get(predicted_stage, "Unknown")

alerts = generate_alerts(
    rollout,
    feature_names,
    context[-1],
    threshold_override=settings["alert_threshold"],
)

# ─── Header Metrics ──────────────────────────────────────────────────────────
col_alert, col_stage, col_count, col_conf = st.columns(4)
col_alert.metric("Threat Level", f"{current_threat:.1%}")
col_stage.metric("Predicted Stage", predicted_stage_name)
col_count.metric("Active Alerts", str(len(alerts)))

if settings["model_type"] == "transformer":
    col_conf.metric("Attention Mode", "Transformer (Multi-Head)")
else:
    col_conf.metric("Attention Mode", "LSTM (Uniform Fallback)")

status_class = "status-danger" if current_threat >= 0.6 else ("status-warning" if current_threat >= 0.3 else "status-safe")
status_text = "ATTACK DETECTED" if current_threat >= 0.6 else ("MONITORING" if current_threat >= 0.3 else "NETWORK SAFE")

st.markdown(
    f'<div style="margin: 8px 0;"><span class="status-badge {status_class}">{status_text}</span></div>',
    unsafe_allow_html=True,
)

# ─── Live Monitoring Charts ──────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🚨 Live Monitoring")
col_km, col_chain = st.columns([1, 1.6])

with col_km:
    gauge = create_gauge_chart(current_threat, title="Infiltration Probability")
    st.plotly_chart(gauge, use_container_width=True)

with col_chain:
    col_chain.markdown("#### 🔗 Predicted Attack Progression")
    stage_at_peak = pred_stages[best_step]
    col_chain.markdown(create_kill_chain_html(int(stage_at_peak)), unsafe_allow_html=True)
    col_chain.markdown(
        f'<p style="color: rgba(255,255,255,0.6); font-size: 0.8rem;">'
        f'Predicted sequence: '
        f'{" → ".join(STAGE_NAMES.get(int(s), "?") for s in dict.fromkeys(pred_stages.tolist()))}</p>',
        unsafe_allow_html=True,
    )

st.plotly_chart(
    create_infiltration_timeline(probs.tolist(), pred_stages.tolist(), title="Infiltration Probability Forecast"),
    use_container_width=True,
)

left, right = st.columns(2)
with left:
    st.plotly_chart(
        create_attack_stage_timeline(pred_stages.tolist(), None, title="Predicted Attack Stage Timeline"),
        use_container_width=True,
    )
with right:
    attn = None
    if attention is not None and attention.get("layer_attention"):
        layer_attn = attention["layer_attention"]
        last_layer = max(layer_attn.keys())
        arr = layer_attn[last_layer]
        arr = arr[0] if arr.ndim == 4 else arr
        attn = arr.mean(axis=0) if arr.ndim == 3 else arr
    if attn is not None:
        st.plotly_chart(create_attention_heatmap(attn, title="Temporal Attention (Last Layer)"),
                        use_container_width=True)
    else:
        st.info("Attention weights are only available for the transformer model.")

# ─── Network State Heatmap ───────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🧬 Network State Evolution")
if states is not None and len(states) >= context_length:
    window_start = min(max(0, context_index), max(0, len(states) - context_length))
    slice_states = np.asarray(states[window_start:window_start + context_length])
    st.plotly_chart(
        create_network_state_heatmap(slice_states, feature_names, title="Observed State History"),
        use_container_width=True,
    )
else:
    st.caption("No raw state timeline available for the selected source.")

# ─── Alerts Feed ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### ⚠️ Alert Feed")

if not alerts:
    st.markdown(
        '<div class="alert-low" style="text-align:center;">No alerts triggered — traffic appears benign.</div>',
        unsafe_allow_html=True,
    )
else:
    for alert in alerts:
        sev = alert.severity
        color = SEVERITY_COLORS.get(sev, "#00ffaa")
        css_class = f"alert-{sev.lower()}"
        actions = " ".join(f'<span style="color:#00aaff;">• {a}</span>' for a in alert.recommended_actions[:3])

        contrib = " ".join(
            f'<span style="color: rgba(255,255,255,0.55); font-size:0.75rem; font-family: JetBrains Mono;">'
            f'{f}: {v:.3f}</span>' for f, v in alert.contributing_features[:4]
        )

        st.markdown(f"""
        <div class="{css_class}">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="color: {color}; font-weight: 700; letter-spacing: 1px;">[{sev}]</span>
                <span style="color: rgba(255,255,255,0.7); font-size: 0.85rem;">
                    {alert.timestamp} — {alert.stage_name}
                </span>
                <span style="color: rgba(255,255,255,0.7); font-size: 0.85rem;">
                    P = {alert.probability:.2%}
                </span>
            </div>
            <div style="margin-top: 8px; display: flex; gap: 8px; flex-wrap: wrap;">{contrib}</div>
            <div style="margin-top: 6px; font-size: 0.85rem;">{actions}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    f'<div style="text-align:center; color: rgba(255,255,255,0.35); font-size: 0.75rem;">'
    f'Model: {settings["model_type"].title()} · Window {context_index + 1}/{n_contexts} · '
    f'Horizon: {settings["k_steps"]} steps · Threshold: {settings["alert_threshold"]:.2f}'
    f'</div>',
    unsafe_allow_html=True,
)