"""
NetSeer — Analysis
==================
Deep-dive analysis of the forecast: predicted state trajectories,
explainability (temporal attention + SHAP) and flow inspection.
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
from session import get_config, get_data, load_model, run_rollout, run_attention, run_shap
from components.charts import (
    create_state_trajectory_chart,
    create_infiltration_timeline,
    create_attack_stage_timeline,
    create_kill_chain_html,
    STAGE_NAMES,
)
from components.explainability_panel import render_explainability_panel, render_mitre_attack_panel

setup_page("NetSeer", "Forecast & Explainability Analysis")

st.markdown("### 🔍 Analysis")
st.markdown(
    "<p style='color: rgba(255,255,255,0.6); font-size: 0.9rem;'>"
    "Inspect the World Model's predicted state trajectory, the temporal attention "
    "it pays across the context window, and SHAP attributions for the current forecast."
    "</p>",
    unsafe_allow_html=True,
)

settings = render_sidebar()

config = get_config()
context_length = int(config["data"]["context_length"])

if settings["data_source"] == "Upload File" and settings["uploaded_file"] is None:
    st.info("👈 Upload data (or use Synthetic Demo) in the sidebar to begin.")
    st.stop()

data = get_data(settings, context_length)
if data is None:
    st.stop()

contexts, seq_stages, feature_names, states, timeline_stages = data
model = load_model(settings["model_type"], settings["model_path"])
if model is None:
    st.error("No trained model weights found. Run `python scripts/train.py` "
             "before using the dashboard, or select a valid weights path in the sidebar.")
    st.stop()

n_contexts = len(contexts)
if n_contexts == 0:
    st.error("No context windows available.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Analysis Controls")
context_index = st.sidebar.slider(
    "Context Window Index",
    min_value=0,
    max_value=n_contexts - 1,
    value=min(int(n_contexts * 0.75), n_contexts - 1),
)
run_deep = st.sidebar.button("🔬 Run Deep Explainability (SHAP)", type="primary")

context = np.asarray(contexts[context_index])
true_stage = int(seq_stages[context_index])

with st.spinner("Simulating future network states..."):
    rollout = run_rollout(model, context, settings["k_steps"])
    attention = run_attention(model, context)

if rollout is None:
    st.stop()

probs = np.asarray(rollout["probabilities"])
pred_stages = np.asarray(rollout["stages"])
predicted_states = np.asarray(rollout["predicted_states"])
confidences = np.asarray(rollout["confidence"])

predicted_stage = max(set(pred_stages.tolist()), key=pred_stages.tolist().count)
predicted_stage_name = STAGE_NAMES.get(int(predicted_stage), "Unknown")

m1, m2, m3 = st.columns(3)
m1.metric("Peak Infiltration Probability", f"{probs.max():.1%}")
m2.metric("Dominant Predicted Stage", predicted_stage_name)
m3.metric("True Stage (Ground Truth)", STAGE_NAMES.get(true_stage, "Unknown"))

tab_pred, tab_explain, tab_inspect = st.tabs(["📈 Prediction", "🔬 Explainability", "📋 Flow Inspector"])

# ─── Prediction Tab ──────────────────────────────────────────────────────────
with tab_pred:
    st.plotly_chart(
        create_infiltration_timeline(probs.tolist(), pred_stages.tolist(), title="Infiltration Forecast"),
        use_container_width=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            create_attack_stage_timeline(pred_stages.tolist(), None, title="Predicted Stages"),
            use_container_width=True,
        )
    with c2:
        st.markdown("#### 🔗 Kill Chain", unsafe_allow_html=True)
        peak_step = int(np.argmax(probs))
        st.markdown(create_kill_chain_html(int(pred_stages[peak_step])), unsafe_allow_html=True)
        st.markdown(
            "<p style='color: rgba(255,255,255,0.6); font-size: 0.85rem; margin-top: 12px;'>"
            "The kill chain shows the furthest stage the model projects the attacker could "
            "reach within the forecast horizon.</p>",
            unsafe_allow_html=True,
        )

    st.plotly_chart(
        create_state_trajectory_chart(
            predicted_states,
            feature_names,
            title="Predicted State Trajectory (Selected Features)",
        ),
        use_container_width=True,
    )

    with st.expander("📊 Raw Forecast Values"):
        import pandas as pd
        df = pd.DataFrame({
            "Step": list(range(len(probs))),
            "Infiltration Probability": probs,
            "Stage": [STAGE_NAMES.get(int(s), "?") for s in pred_stages],
            "Confidence": confidences,
        })
        st.dataframe(df, use_container_width=True, height=260)

# ─── Explainability Tab ──────────────────────────────────────────────────────
with tab_explain:
    attn = None
    if attention is not None and attention.get("layer_attention"):
        all_layers = [np.asarray(a) for a in attention["layer_attention"].values()]
        avg = np.mean(all_layers, axis=0)          # [batch, heads, seq, seq]
        attn = avg[0].mean(axis=0) if avg.ndim == 4 else avg.mean(axis=0)

    shap_vals = None
    top_features = None
    if run_deep:
        with st.spinner("Computing SHAP attributions (gradient-based)..."):
            shap_out = run_shap(
                model, contexts, context, feature_names,
                method="gradient", background_n=8,
            )
            if shap_out is not None:
                sv = np.asarray(shap_out["result"]["shap_values"])
                top_features = shap_out["top_features"]
                if sv.ndim == 4:
                    shap_vals = sv[0]  # [seq, features, outputs] for panel aggregation
                else:
                    shap_vals = sv

    if not run_deep:
        st.info("Click **Run Deep Explainability (SHAP)** in the sidebar to compute feature attributions.")

    render_explainability_panel(
        shap_values=shap_vals if shap_vals is not None else None,
        attention_weights=attn,
        feature_names=feature_names,
        top_features=top_features,
        stage_name=predicted_stage_name,
        stage_idx=int(predicted_stage),
    )

    st.markdown("---")
    st.markdown("### 🎯 Context Window Stage Info")

    from src.prediction.mitre_mapper import MitreMapper
    mapper = MitreMapper()
    stage_info = mapper.get_stage_info(true_stage)
    render_mitre_attack_panel(true_stage, stage_info)

# ─── Flow Inspector ──────────────────────────────────────────────────────────
with tab_inspect:
    st.markdown(
        "<p style='color: rgba(255,255,255,0.7); font-size: 0.9rem;'>"
        f"Raw values of the {context_length} observed windows (context #{context_index}). "
        "Each row is a time step; columns are latent network state features."
        "</p>",
        unsafe_allow_html=True,
    )

    import pandas as pd
    ctx_df = pd.DataFrame(context, columns=feature_names)
    top_cols = [f for f in feature_names[:20]]
    st.dataframe(
        ctx_df[top_cols].T,
        use_container_width=True,
        height=420,
    )