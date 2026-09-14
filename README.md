# NetSeer — AI-Based Network Attack Forecasting

NetSeer is an offline-first World Model system that forecasts how an attacker will
progress across the MITRE ATT&CK kill chain. It ingests network flow telemetry
(CIC-IDS-2018 CSVs, PCAP captures, or pre-built state files), encodes the recent
network history into latent state windows, and uses a **temporal transformer World Model**
to simulate the next *K* steps, estimate infiltration probability, classify the predicted
attack stage, and surface severity-graded alerts with SHAP + attention explainability.

![Stack](https://img.shields.io/badge/stack-PyTorch%20%7C%20Streamlit%20%7C%20Plotly-00ffaa)

---

## Quick Start

```bash
# 1. Install dependencies (Python 3.10+)
pip install -r requirements.txt

# 2. Generate the synthetic demo dataset
python scripts/generate_synthetic.py --num_windows 1000

# 3. Train the World Model (CPU-friendly)
python scripts/train.py --config config.yaml \
    --data_source synthetic --model_type transformer --epochs 40 --output_dir models

# 4. Benchmark against baselines (LSTM, Logistic Regression)
python scripts/evaluate.py --config config.yaml \
    --model_path models/best_model.pth --data_source synthetic \
    --data_dir data/synthetic --output_dir results

# 5. Launch the dashboard
streamlit run app/app.py
```

> The repository ships with a trained transformer checkpoint (`models/transformer_model.pth`),
> LSTM baseline (`models/lstm_model.pth`), regenerated synthetic data and benchmark results so
> the dashboard works immediately after `pip install`.

## For a non-technical audience (one-click console)

The home page (`streamlit run app/app.py`) is a friendly console with three buttons —
no CIDR maths, no sliders required:

1. **🛰️ Scan my Wi-Fi** — auto-detects the network you're connected to right now, finds the
   devices, checks which doors are open, then asks the AI what an attacker would do next.
   Safe (LOW/NONE risk) networks get a green verdict with practical tips only — no alarming
   percentages. Exposed networks get the full AI attack-path forecast in plain words.
   Each open door also comes with a **close-guide** (what it is, why it's risky, how to turn it
   off), plus a "🧪 see this network with those doors closed" button that safely simulates the
   tightened version before you touch anything.
   Advanced controls (interface, port list, timeout) are tucked behind an expander.
2. **🧪 Simulated attack** — plays a realistic attack against a *fake* company network behind
   the scenes (nothing touches the wire), then narrates what happened and shows how the AI's
   confidence rose as the attack played out. Great for a judges' demo with zero network setup.
3. **🔭 Live watch** — leaves the AI running on your current network; a rolling alarm fires in
   plain words if an attack pattern starts ("🚨 An attack pattern is starting — AI confidence 87%").

The artificial intelligence is the same pipeline everywhere; the console just hides the
technical plumbing. Power users keep the full dashboard, Network Scout and Live Monitor pages
in the sidebar.

## Live Traffic + Network Scanning

Two pages talk to a **real network** (for own-hotspot / lab demos):

- **📡 Live Monitor** — captures traffic on a live interface, maps each 5 s window into
  76-dim live states (`src/data/live_capture.py`), and runs the K-step forecast. Uses the
  live-trained checkpoint (`models/best_live_model.pth`, from `scripts/train_live.py`).
- **🛰️ Network Scout** — the "connect to a network, analyse it, predict threats" page:
  1. Enumerates hosts on the target subnet (ARP sweep when Npcap is installed, ICMP `ping`
     sweep otherwise — no install required).
  2. Runs a light TCP connect scan on curated common ports.
  3. Scores each host's attack surface and maps it to MITRE ATT&CK techniques (T1046/T1190/T1078).
  4. Replays the scanner's **own** SYN probes through the flow → state → world-model pipeline
     (`emit_probe_flows`) so the forecast reacts to the reconnaissance it just performed
     (0 probes sent twice over the wire).

CLI equivalent:

```bash
python scripts/scan_network.py --network 192.168.1.0/24 --ports 22,80,443,445,3389 --forecast
# optional: --method {auto,arp,icmp) · --timeout 0.8 · --json out.json
```

**Npcap (optional):** only ARP host discovery needs it. Without Npcap the Scout falls back to
ICMP echo discovery and the Live Monitor to whatever interface state is exposed; the scanner,
risk scoring and world-model forecast still work. Install from https://npcap.com/#download if you
want ARP/MAC enumeration and full packet capture.

## Dashboard

`streamlit run app/app.py` opens a dark-themed multi-page app:

| Page | Purpose |
|------|---------|
| **Home (console)** | One-click friendly entry: scan my Wi-Fi · simulated attack · live watch |
| **📡 Live Monitor** | Live capture → rolling K-step forecast, interface selector, live-trained World Model |
| **🛰️ Network Scout** | Real-network host discovery + port scan → attack-surface scoring → world-model forecast on the scan |
| **Dashboard** | Static data — threat gauge, infiltration timeline, predicted stage timeline, temporal-attention heatmap, network-state heatmap, alert feed |
| **Analysis** | Deep dive — predicted state trajectory, kill chain, temporal attention, SHAP feature attributions, MITRE ATT&CK mapping per stage |
| **Benchmarks** | Model comparison table, ROC curves, per-stage confusion matrices, training curves |
| **About** | Architecture, MITRE mapping, dataset and usage documentation |

Data sources (static-data pages): **Synthetic Demo** data, an **uploaded** CSV (CIC-IDS-2018 flow
format or NetSeer state format) / PCAP, or a **directory** of CSVs.

## Pipeline

```
Raw traffic (PCAP / CIC-IDS CSV)
   │  src/data/pcap_parser.py · csv_loader.py
   ▼
Flow features (68 CIC-IDS features)          src/data/feature_extractor.py
   ▼
Normalize (z-score)                          src/data/normalizer.py
   ▼
Temporal windows (60 s) → state vectors      src/data/state_builder.py
   ▼
PCA compression → 76-dim state               src/data/state_builder.compress_states
   ▼
Rolling context windows (16 states)
   ▼
World Model (temporal transformer)           src/models/world_model.py
   ├─ Transition head (MDN, 5 Gaussians)     next-state prediction
   ├─ Infiltration head                       P(attack in next state)
   └─ Stage head                             6-way ATT&CK stage classification
   ▼
K-step autoregressive rollout                 src/prediction/rollout_engine.py
   ▼
Alerts · MITRE mapping · Explainability       src/prediction/alert_generator.py
                                             src/prediction/mitre_mapper.py
                                             src/explainability/*.py
```

## Project Layout

```
├── app/                  Streamlit dashboard
│   ├── app.py            One-click console (scan / simulate / live watch) + landing
│   ├── theme.py          Shared dark theme + page setup
│   ├── session.py        Data/model/inference helpers for pages
│   ├── pages/            Live Monitor · Network Scout · Dashboard · Analysis · Benchmarks · About
│   └── components/       charts.py · sidebar.py · explainability_panel.py
├── src/
│   ├── data/             parsers, feature extraction, state building, synthetic generator,
│   │                     live capture engine, active network scanner
│   ├── models/           world model, temporal transformer, LSTM & logistic baselines
│   ├── prediction/       rollout engine, alert generator, MITRE mapper, attack simulator
│   │                     (dummy-network attack playback + plain-language reporting)
│   ├── explainability/   attention + SHAP explainers, feature reports
│   └── utils/            config, device, seeding
├── scripts/
│   ├── train.py          training entrypoint
│   ├── train_live.py     train on live-style flow states -> models/best_live_model.pth
│   ├── scan_network.py   CLI network scanner + world-model forecast
│   ├── evaluate.py       benchmark harness (metrics + ROC + confusion matrices)
│   ├── generate_synthetic.py
│   └── download_data.py  optional CIC-IDS-2018 downloader
├── data/synthetic/       synthetic demo dataset + dashboard CSV
├── models/               trained checkpoints + training histories
├── results/              benchmark_results.json
├── tests/                unittest suite
└── config.yaml           all pipeline/model/training configuration
```

## Testing

The test suite is `unittest`-based (pytest is installed as a runner but not required):

```bash
python -m unittest discover -s tests -v
# or
pytest -q
```

## Configuration

Everything is parameterised through `config.yaml`:

- `data.context_length` — context windows fed to the model (16)
- `features.num_state_features` — compressed state dimension (76)
- `model.*` — transformer depth/width, MDN mixture count, attack-stage count
- `training.*` — optimizer, LR schedule, loss weights
- `prediction.*` — K-step horizon, alert thresholds
- `explainability.*` — SHAP method, background sample count

## Benchmarks

Results are saved to `results/benchmark_results.json` by `scripts/evaluate.py` and rendered on
the **Benchmarks** page. On the synthetic dataset the World Model and LSTM both reach ~1.0
metrics because overlapping sliding windows are highly self-similar — treat these as
*validation of the pipeline*, not real-world accuracy. Use the CIC-IDS-2018 dataset
(`scripts/download_data.py`) for production-grade reporting.

## Notes

- The attention weights shown in the dashboard are the **real** per-head attention matrices
  captured by the custom `TemporalEncoderLayer` (torch ≥ 2.13 fuses the built-in
  `nn.TransformerEncoderLayer` fast path and hides attention at the Python level).
- **scapy is pinned to 2.6.1** in `requirements.txt` — Scapy 2.7 broke the classic packet API
  (`pkt.haslayer`, `IP in pkt`) this codebase relies on.
- Runs fully offline on CPU by default (`training.device: auto`).
- Research prototype; probabilistic simulations are not guarantees of intrusion.