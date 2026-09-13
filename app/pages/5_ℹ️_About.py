"""
NetSeer — About
===============
Project documentation: architecture, MITRE mapping, datasets and usage guide.
"""

import sys
import os

_PAGES_DIR = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.dirname(_PAGES_DIR)
_PROJECT_ROOT = os.path.dirname(_APP_DIR)
for _p in (_APP_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st

from theme import setup_page
from src.prediction.mitre_mapper import MitreMapper
from components.charts import STAGE_COLORS

setup_page("NetSeer", "Documentation & Methodology")

st.markdown("### ℹ️ About NetSeer")
st.markdown(
    "<p style='color: rgba(255,255,255,0.7); font-size: 0.95rem; line-height: 1.7;'>"
    "**NetSeer** is an AI-based network attack forecasting system. It ingests network flow "
    "telemetry, builds a latent representation of the network state over rolling time "
    "windows, and uses a **World Model** — a temporal transformer trained to predict "
    "<code>P(State<sub>t+1</sub> | State<sub>t-k+1..t</sub>)</code> — to simulate the next "
    "<em>K</em> steps of the attack, estimate infiltration probability, classify the predicted "
    "attack stage and surface actionable alerts mapped to the MITRE ATT&CK framework."
    "</p>",
    unsafe_allow_html=True,
)

# ─── Architecture ────────────────────────────────────────────────────────────
st.markdown("### 🏗️ System Architecture")

st.markdown("""
```
┌─────────────┐   ┌──────────────┐   ┌──────────────┐   ┌───────────────┐   ┌──────────────┐
│ PCAP / CSV  │──▶│  Feature     │──▶│  State       │──▶│  World Model  │──▶│  Rollout     │
│ (CIC-IDS)   │   │  Extraction  │   │  Builder     │   │  (Temporal    │   │  Engine (K)  │
└─────────────┘   └──────────────┘   └──────────────┘   │  Transformer) │   └──────┬───────┘
                                                        └───────────────┘          │
                                    ┌───────────────┐   ┌──────────────────┐   ┌─────▼─────┐
                                    │  Alerts ·     │◀──│  Infiltration    │◀──│ Stage     │
                                    │  Mitre Map    │   │  Probability     │   │ Forecast  │
                                    └───────────────┘   └──────────────────┘   └───────────┘
                                    ┌───────────────┐   ┌──────────────────┐
                                    │  Explainability│   │  SHAP · Attention│
                                    └───────────────┘   └──────────────────┘
```
""")

st.markdown("**Pipeline components**")

comps = st.columns(3)
comps[0].markdown("""
**Data → State**
- `PcapParser` / `CICIDSLoader` parse raw traffic into flows
- `FeatureExtractor` (68 CIC-IDS features)
- `FeatureNormalizer` + `StateBuilder`
  (60 s temporal windows, rolling context)
- `compress_states` PCA → 76-dim latent state vector
""")
comps[1].markdown("""
**World Model**
- Temporal encoder: 3 layers × 4 heads, `latent_dim=128`, `ffn=256`
- Transition head (MDN with 5 Gaussian components)
- Infiltration head (binary probability)
- Stage head (6-way ATT&CK stage classification)
- Trained with weighted multi-task loss
""")
comps[2].markdown("""
**Prediction / Explainability**
- `RolloutEngine`: K-step autoregressive forward simulation
- `AlertGenerator`: severity-banded alerts (LOW→CRITICAL)
- `MitreMapper`: stage → ATT&CK tactic & techniques
- `AttentionExplainer` + `ShapExplainer`: per-prediction reasoning
""")

# ─── MITRE Mapping ───────────────────────────────────────────────────────────
st.markdown("### 🎯 MITRE ATT&CK Stage Mapping")

mapper = MitreMapper()
stage_rows = []
for i, info in enumerate(mapper.get_all_stages()):
    color = STAGE_COLORS.get(i, "#888")
    tech = ", ".join(info["technique_ids"]) if info["technique_ids"] else "—"
    stage_rows.append(
        f'<tr><td style="padding:10px;"><span style="width:10px;height:10px;border-radius:50%;'
        f'display:inline-block;background:{color};box-shadow:0 0 8px {color};margin-right:8px;"></span>'
        f'<b>{info["name"]}</b></td>'
        f'<td style="padding:10px;color:rgba(255,255,255,0.7);">{info["description"]}</td>'
        f'<td style="padding:10px;color:#00aaff;font-family:JetBrains Mono;">{tech}</td></tr>'
    )

st.markdown(f"""
<table style="width:100%; border-collapse:collapse; font-size:0.85rem;">
    <thead>
        <tr style="border-bottom:1px solid rgba(0,255,170,0.2);">
            <th style="padding:10px;text-align:left;color:#00ffaa;">Stage</th>
            <th style="padding:10px;text-align:left;color:#00ffaa;">Description</th>
            <th style="padding:10px;text-align:left;color:#00ffaa;">ATT&CK Techniques</th>
        </tr>
    </thead>
    <tbody>{''.join(stage_rows)}</tbody>
</table>
""", unsafe_allow_html=True)

# ─── Datasets ────────────────────────────────────────────────────────────────
st.markdown("### 🗄️ Datasets")

d1, d2 = st.columns(2)
with d1:
    st.markdown("""
**Synthetic Demo Dataset** *(used by the dashboard)*
- 1,000 temporal windows generated by `scripts/generate_synthetic.py`
- 76-dim state vectors, sequence length 16 windows
- Built-in infiltration stage distribution: 30.5%
- Ground-truth next-state and attack-stage labels included
- File: `data/synthetic/synthetic_data.npz`
""")
with d2:
    st.markdown("""
**CIC-IDS-2018** *(optional, production)*
- 80 real attack flows over 10 days, 6 attack categories
- Download via `scripts/download_data.py` into `data/raw`
- Requires `context_length × 60 s` of data per analysis window
- Format accepted automatically by the dashboard (CSV / PCAP)
""")

# ─── Usage ───────────────────────────────────────────────────────────────────
st.markdown("### 🚀 Usage Guide")

st.markdown("""
1. **Install** — `pip install -r requirements.txt`
2. **Generate demo data** — `python scripts/generate_synthetic.py --num_windows 1000`
3. **Train** — `python scripts/train.py --config config.yaml --data_source synthetic --model_type transformer`
4. **Benchmark** — `python scripts/evaluate.py --config config.yaml --model_path models/best_model.pth`
5. **Launch dashboard** — `streamlit run app/app.py`

**Analyze your own traffic**: from the sidebar choose *Upload File* and provide a
CIC-IDS-2018 CSV, a PCAP capture, or a NetSeer state CSV (76 numeric columns).
""")

# ─── Tech Stack ──────────────────────────────────────────────────────────────
st.markdown("### 🧰 Technology Stack")

stack = st.columns(4)
stack[0].markdown("**Core ML**\n- PyTorch\n- NumPy / pandas\n- scikit-learn\n- SHAP")
stack[1].markdown("**Networking**\n- Scapy (PCAP)\n- CIC-IDS-2018 parser\n- MITRE ATT&CK")
stack[2].markdown("**Visualization**\n- Streamlit\n- Plotly\n- Custom dark theme")
stack[3].markdown("**Quality**\n- unittest suite\n- Evaluation harness\n- Seeded experiments")

st.markdown("---")
st.markdown(
    '<div style="text-align:center; color: rgba(255,255,255,0.35); font-size: 0.8rem;">'
    "NetSeer v1.0 — Research prototype for academic / evaluation use only. "
    "Forecasts are probabilistic simulations, not guarantees of intrusion."
    "</div>",
    unsafe_allow_html=True,
)