"""
NetSeer — Benchmarks
====================
Model evaluation results: metric comparison, ROC curves, confusion
matrices and training curves for the World Model vs LSTM vs Logistic baselines.
"""

import sys
import os
import subprocess

_PAGES_DIR = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.dirname(_PAGES_DIR)
_PROJECT_ROOT = os.path.dirname(_APP_DIR)
for _p in (_APP_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st

from theme import setup_page
from session import load_benchmarks, load_history
from components.charts import (
    create_metric_comparison_table,
    create_roc_curve,
    create_confusion_matrix,
    create_training_history,
    STAGE_NAMES,
)

setup_page("NetSeer", "Model Benchmarks & Evaluation")

st.markdown("### 📈 Benchmarks")
st.markdown(
    "<p style='color: rgba(255,255,255,0.6); font-size: 0.9rem;'>"
    "Benchmark results are produced offline by <code>scripts/evaluate.py</code> on the "
    "synthetic dataset. Run the benchmark any time the models or data change."
    "</p>",
    unsafe_allow_html=True,
)

benchmarks = load_benchmarks()

col_run = st.columns([1, 2])[0]
if col_run.button("🔄 Re-run Benchmark (scripts/evaluate.py)", type="secondary"):
    py = os.path.join(PROJECT_ROOT, ".venv", "Scripts", "python.exe")
    script = os.path.join(PROJECT_ROOT, "scripts", "evaluate.py")
    cmd = [py, "-W", "ignore::UserWarning", script]
    with st.spinner("Running benchmark (this may take a minute)..."):
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=1800)
    if proc.returncode == 0:
        st.success("Benchmark completed. Results refreshed.")
    else:
        st.error(f"Benchmark failed:\n{proc.stderr[-2000:]}")

benchmarks = load_benchmarks()

if benchmarks is None:
    st.warning("No benchmark results found. Click **Re-run Benchmark** above to generate them.")
    st.stop()

# ─── Model Comparison Table ──────────────────────────────────────────────────
metric_keys = ["F1", "Precision", "Recall", "FPR", "AUC-ROC", "Accuracy"]
model_keys = list(benchmarks.keys())
model_names = [benchmarks[k]["name"] for k in model_keys]

metrics = {mk: [benchmarks[k]["metrics"].get(mk, 0.0) for k in model_keys] for mk in metric_keys}

st.markdown("### 🏆 Model Comparison")
st.plotly_chart(
    create_metric_comparison_table(model_names, metrics, title="Benchmark Comparison Matrix"),
    use_container_width=True,
)

# ─── ROC Curves ──────────────────────────────────────────────────────────────
st.markdown("### 🎯 ROC Curves")

import numpy as np
import plotly.graph_objects as go

from components.charts import COLORS

roc_fig = go.Figure()
roc_fig.add_trace(go.Scatter(
    x=[0, 1], y=[0, 1], mode="lines",
    line=dict(dash="dash", color="rgba(255,255,255,0.3)", width=1),
    name="Random (AUC = 0.5)",
))
curve_colors = [COLORS["primary"], COLORS["secondary"], COLORS["warning"]]
for i, k in enumerate(model_keys):
    m = benchmarks[k]["metrics"]
    fpr = np.asarray(m.get("fpr_curve", [0, 1]))
    tpr = np.asarray(m.get("tpr_curve", [0, 1]))
    auc = m.get("AUC-ROC", 0.0)
    roc_fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines", name=f"{benchmarks[k]['name']} (AUC = {auc:.4f})",
        line=dict(color=curve_colors[i % len(curve_colors)], width=2.5),
    ))

roc_fig.update_layout(
    template="plotly_dark",
    title=dict(text="ROC Comparison", font=dict(color="rgba(255,255,255,0.8)")),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    height=450,
    xaxis=dict(title="False Positive Rate", gridcolor="rgba(255,255,255,0.06)", range=[0, 1]),
    yaxis=dict(title="True Positive Rate", gridcolor="rgba(255,255,255,0.06)", range=[0, 1]),
)
st.plotly_chart(roc_fig, use_container_width=True)

# ─── Confusion Matrices ──────────────────────────────────────────────────────
st.markdown("### 🔢 Confusion Matrices (Per Attack Stage)")
stage_labels = list(STAGE_NAMES.values())

cm_tabs = st.tabs([benchmarks[k]["name"] for k in model_keys])
for tab, k in zip(cm_tabs, model_keys):
    with tab:
        m = benchmarks[k]["metrics"]
        cm = np.asarray(m.get("stage_cm"))
        if cm is not None and cm.size > 0:
            st.plotly_chart(
                create_confusion_matrix(cm, stage_labels, title=f"{benchmarks[k]['name']} — Stage Confusion"),
                use_container_width=True,
            )
            c1, c2 = st.columns(2)
            c1.metric("Stage Accuracy", f"{m.get('Stage Accuracy', 0.0):.4f}")
            c2.metric("Stage F1 (macro)", f"{m.get('Stage F1 (macro)', 0.0):.4f}")
        else:
            st.info("No stage-level evaluation available for this model.")

# ─── Training History ────────────────────────────────────────────────────────
st.markdown("### 📉 Training Curves")

histories = [
    ("transformer", "Temporal Transformer"),
    ("lstm", "LSTM Baseline"),
]
history_tabs = st.tabs([h[1] for h in histories])

for tab, (htype, hname) in zip(history_tabs, histories):
    with tab:
        hist = load_history(htype)
        if hist and any("loss" in k.lower() for k in hist):
            st.plotly_chart(
                create_training_history(hist, title=f"{hname} — Training History"),
                use_container_width=True,
            )
        else:
            st.info(f"No saved training history for {hname}.")

# ─── Notes ───────────────────────────────────────────────────────────────────
with st.expander("📝 How to read these numbers"):
    st.markdown("""
**Dataset**: Synthetic network attack sequences (1,000 windows, `scripts/generate_synthetic.py`).
Metrics are computed with a seeded random train/test split.

- **F1 / Precision / Recall** — detection of *infiltration* (any attack stage vs benign).
- **AUC-ROC** — whole-curve ranking quality for the infiltration score.
- **Stage Confusion** — multi-class classification of the predicted next stage.

> The World Model and LSTM achieve near-perfect scores on this synthetic benchmark because
> overlapping sliding windows are highly self-similar. These results are for pipeline
> validation, not a claim of real-world accuracy — use the real CIC-IDS-2018 dataset
> for production-grade reporting.
    """)