"""
NetSeer — Live Network Monitor
==============================
Streams **real network packets** (live interface, PCAP replay or an on-device
attack simulator) through the same flow → state → World Model → forecast
pipeline used by the offline dashboard, continuously, in real time.

The page shows the World Model making a *stream of decisions* over live
traffic: every finished time-window produces a forecast, alert and MITRE stage
estimate. A live session can also be exported back to CSV for re-training, so
the model can be adapted to the operator's own network (closed loop).
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
from session import get_config, load_model, run_rollout, generate_alerts, available_live_models
from components.charts import (
    create_gauge_chart,
    create_infiltration_timeline,
    create_attack_stage_timeline,
    create_kill_chain_html,
    SEVERITY_COLORS,
    STAGE_NAMES,
)

setup_page("NetSeer", "Live Network Monitor — Real Traffic, Real Decisions")

from src.data.live_capture import (
    LiveTrafficEngine,
    LiveStateMapper,
    OnlineScaler,
    extract_window_states,
    list_interfaces,
)

config = get_config()
CONTEXT_LENGTH = int(config["data"]["context_length"])
STATE_DIM = int(config["features"]["num_state_features"])

FEEDS = ["PCAP Replay", "Live Interface", "Attack Simulator"]

st.markdown("### 📡 Live Network Monitor")
st.markdown(
    "<p style='color: rgba(255,255,255,0.6); font-size: 0.9rem;'>"
    "Watch the World Model analyze <b>raw packets as they arrive</b>. Every "
    "closed time-window is reduced to a 76-dim state and rolled forward to an "
    "infiltration forecast, alert and MITRE stage estimate.</p>",
    unsafe_allow_html=True,
)

running = False
if "live_engine" in st.session_state and st.session_state["live_engine"].running:
    running = True

# ─── Feed configuration (frozen while a session is running) ────────────────
st.markdown("### 🎛️ Capture Source")
feed = st.radio("Traffic feed", FEEDS, index=0, horizontal=True, disabled=running)

replay_file = None
interface = None
sim_kind = "Port Scan"
sim_target = "192.168.1.100"
demo_seconds = 120.0

if feed == "PCAP Replay":
    replay_file = st.file_uploader(
        "PCAP / PCAPNG file",
        type=["pcap", "pcapng"],
        disabled=running,
        help="A capture of real traffic is projected onto a compressed demo clock.",
    )
    demo_seconds = st.slider("Replay duration (simulated seconds)", 10, 300, 120,
                             disabled=running, help="The whole PCAP plays in this much simulated time.")
elif feed == "Live Interface":
    ifaces = list_interfaces()
    if not ifaces:
        st.warning("No capture interfaces detected. Install **Npcap** and run "
                   "with capture permissions, or use one of the other feeds.")
    interface = st.selectbox("Network interface", ifaces, disabled=running) if ifaces else None
    st.caption("Requires Npcap and capture privileges. Live traffic is processed locally.")
elif feed == "Attack Simulator":
    sim_kind = st.selectbox("Simulated attack", ["Port Scan", "SSH Brute-Force"],
                            disabled=running)
    sim_target = st.text_input("Target IP", value=sim_target, disabled=running)
    st.caption("Generates local packets and feeds them through exactly the same pipeline — "
               "no privileges required.")

st.markdown("### 🧠 Analysis Settings")
mc1, mc2, mc3, mc4 = st.columns(4)
_model_choices = available_live_models()
_model_labels = [label for label, _t, _f in _model_choices]
_model_default = next((i for i, (label, t, f) in enumerate(_model_choices) if "Live World" in label), 0)
_sel = mc1.selectbox("Model", _model_labels, index=_model_default, disabled=running)
_model_type = next((t for label, t, f in _model_choices if label == _sel), "transformer")
_model_family = next((f for label, t, f in _model_choices if label == _sel), "offline")
k_steps = mc2.slider("Forecast horizon (steps)", 1, 30, 10, disabled=running)
threshold = mc3.slider("Alert threshold", 0.1, 0.95, 0.6, step=0.05, disabled=running)
window_size = mc4.slider("Window size (s)", 2, 60, 5, disabled=running,
                         help="Seconds of traffic per state. Smaller = faster reaction.")

# ─── Start / Stop ───────────────────────────────────────────────────────────
if running:
    st.info(f"⏺ Capturing live from **{st.session_state['live_engine'].source}** — "
            "monitoring below updates automatically.")
    if st.button("🛑 Stop Capture", type="secondary"):
        st.session_state["live_engine"].stop()
        st.rerun()
else:
    can_start = True
    if feed == "PCAP Replay" and replay_file is None:
        can_start = False
    if feed == "Live Interface" and not interface:
        can_start = False
    if st.button("▶️ Start Live Monitoring", type="primary", disabled=not can_start):
        engine = LiveTrafficEngine()
        if feed == "PCAP Replay":
            import tempfile
            bytes_data = replay_file.getvalue()
            ext = ".pcapng" if replay_file.name.lower().endswith(".pcapng") else ".pcap"
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp.write(bytes_data)
                tmp_path = tmp.name
            ok = engine.start_replay(tmp_path, demo_seconds=float(demo_seconds))
            os.remove(tmp_path)
            if not ok:
                st.error(engine.error or "Failed to start replay.")
        elif feed == "Live Interface":
            engine.start_interface(interface)
        else:
            if sim_kind == "SSH Brute-Force":
                engine.inject_bruteforce(target=sim_target)
            else:
                engine.inject_scan(target=sim_target)

        st.session_state["live_engine"] = engine
        st.session_state["live_runtime"] = {
            "mapper": LiveStateMapper(),
            "scaler": OnlineScaler(STATE_DIM),
            "states": [],
            "next_window": 0,
            "history": [],
            "last": None,
            "last_state": None,
            "window_size": float(window_size),
            "k_steps": int(k_steps),
            "threshold": float(threshold),
            "context_length": CONTEXT_LENGTH,
            "model_type": _model_type,
            "model_family": _model_family,
            "state_dim": STATE_DIM,
        }
        st.rerun()

# ─── Live monitor (auto-refreshes every 2 s while a session runs) ───────────
@st.fragment(run_every=(2.0 if running else None))
def render_monitor():
    engine = st.session_state.get("live_engine")
    rt = st.session_state.get("live_runtime")

    # Session had ended or never started.
    if engine is None or rt is None:
        st.markdown(
            "<p style='color: rgba(255,255,255,0.4);'>Choose a capture source above and press "
            "**Start Live Monitoring**.</p>", unsafe_allow_html=True)
        return
    if not engine.running:
        status = engine.status()
        if status.get("error"):
            st.error(f"Capture stopped: {status['error']}")
        else:
            st.warning(f"Capture ended — {status['packets']:,} packets · "
                       f"{status['flows']:,} flows · {len(rt['states'])} windows analyzed "
                       f"from `{status['source']}`.")
        if st.button("🗑️ Clear session"):
            for key in ("live_engine", "live_runtime"):
                st.session_state.pop(key, None)
            st.rerun()
        return

    model = load_model(rt["model_type"], family=rt.get("model_family", "live"))
    if model is None:
        st.error("No trained model weights found. Run `python scripts/train.py` "
                 "or `python scripts/train_live.py` first.")
        return

    status = engine.status()
    flows = engine.snapshot_flows()

    # ── ingest new completed windows → states ──
    if flows is not None and not flows.empty and status.get("sim_t0"):
        new_states, rt["next_window"] = extract_window_states(
            flows, float(status["sim_t0"]), rt["window_size"],
            rt["mapper"], rt["scaler"], rt["next_window"],
        )
        for vec in new_states:
            rt["states"].append(vec)
            if len(rt["states"]) > 512:
                rt["states"] = rt["states"][-512:]

    # ── rolling forecast over the latest context ──
    latest = None
    if len(rt["states"]) >= rt["context_length"]:
        ctx = np.asarray(rt["states"][-rt["context_length"]:], dtype=np.float32)
        rollout = run_rollout(model, ctx, rt["k_steps"])
        if rollout is not None:
            probs = np.asarray(rollout["probabilities"])
            stages = np.asarray(rollout["stages"])
            best = int(np.argmax(probs))
            latest = {
                "prob": float(probs[best]),
                "prob_peak": float(probs.max()),
                "stage": int(stages[best]),
                "stage_seq": [int(s) for s in stages],
                "prob_seq": probs.tolist(),
                "conf_seq": np.asarray(rollout["confidence"]).tolist(),
                "rollout": rollout,
            }
            rt["last"] = latest
            rt["last_state"] = ctx[-1]

    # history point for every finished window (or every tick once resolving)
    if len(rt["states"]) >= rt["context_length"]:
        h = rt["last"]
        if h is not None:
            rt["history"].append({
                "t": f"W{len(rt['states'])}",
                "prob": h["prob_peak"],
                "stage": h["stage"],
            })
            if len(rt["history"]) > 300:
                rt["history"] = rt["history"][-300:]

    # ── status row ──
    col_a, col_b, col_c, col_d, col_e = st.columns(5)
    col_a.metric("Feed", status["mode"].title())
    col_b.metric("Packets", f"{status['packets']:,}")
    col_c.metric("Flows", f"{status['flows']:,}")
    col_d.metric("Windows", len(rt["states"]))
    col_e.metric("Forecast", "Live" if rt["last"] is not None else "Warm-up")

    # ── warm-up progress ──
    n_warm = rt["context_length"]
    warm = min(len(rt["states"]) / n_warm, 1.0)
    if rt["last"] is None:
        st.progress(warm, text=f"Collecting context — {len(rt['states'])}/{n_warm} windows")
    else:
        st.caption(f"Context windows: {len(rt['states'])}  ·  "
                   f"state dim: {rt['state_dim']}  ·  window: {rt['window_size']:.0f}s")

    if rt["last"] is None:
        st.markdown(
            "<p style='color: rgba(255,255,255,0.5);'>Collecting enough traffic to form a context "
            "window. Use shorter windows, a PCAP replay, or the attacker simulator to converge faster.</p>",
            unsafe_allow_html=True)
        return

    latest = rt["last"]
    pred_stage = latest["stage"]
    peak = latest["prob_peak"]

    # ── header metrics ──
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Peak Infiltration Probability", f"{peak:.1%}")
    m2.metric("Predicted Stage", STAGE_NAMES.get(pred_stage, "Unknown"))
    alerts = generate_alerts(latest["rollout"], [f"feature_{i}" for i in range(rt["state_dim"])],
                             rt["last_state"], threshold_override=rt["threshold"])
    m3.metric("Active Alerts", len(alerts))
    m4.metric("Packets / Window", f"{(status['packets'] / max(len(rt['states']), 1)):.0f}")

    status_class = "status-danger" if peak >= 0.6 else ("status-warning" if peak >= 0.3 else "status-safe")
    status_text = "ATTACK DETECTED" if peak >= 0.6 else ("MONITORING" if peak >= 0.3 else "NETWORK SAFE")
    st.markdown(
        f'<div style="margin: 8px 0;"><span class="status-badge {status_class} threat-active">'
        f'{status_text}</span></div>', unsafe_allow_html=True)

    # ── forecast charts ──
    gc, kc = st.columns([1, 1.6])
    with gc:
        st.plotly_chart(create_gauge_chart(peak, title="Infiltration Probability"),
                        use_container_width=True)
    with kc:
        st.markdown("#### 🔗 Predicted Attack Progression")
        st.markdown(create_kill_chain_html(pred_stage), unsafe_allow_html=True)
        seq = " → ".join(STAGE_NAMES.get(int(s), "?") for s in latest["stage_seq"])
        st.markdown(f'<p style="color: rgba(255,255,255,0.6); font-size: 0.8rem;">'
                    f'Forecast sequence: {seq}</p>', unsafe_allow_html=True)

    st.plotly_chart(
        create_infiltration_timeline(latest["prob_seq"], latest["stage_seq"],
                                     title=f"Rolling Forecast — latest window (K={rt['k_steps']})"),
        use_container_width=True,
    )

    col_h1, col_h2 = st.columns(2)
    with col_h1:
        hist_probs = [p["prob"] for p in rt["history"]]
        hist_stages = [p["stage"] for p in rt["history"]]
        st.plotly_chart(
            create_infiltration_timeline(hist_probs, hist_stages,
                                         title="Decision History (every finished window)"),
            use_container_width=True,
        )
    with col_h2:
        st.plotly_chart(
            create_attack_stage_timeline(hist_stages,
                                         title="Predicted Stages Over Time"),
            use_container_width=True,
        )

    # ── alert feed ──
    st.markdown("### ⚠️ Alert Feed")
    if not alerts:
        st.markdown(
            '<div class="alert-low" style="text-align:center;">No alerts triggered — traffic appears benign.</div>',
            unsafe_allow_html=True)
    else:
        for alert in alerts:
            sev = alert.severity
            color = SEVERITY_COLORS.get(sev, "#00ffaa")
            css_class = f"alert-{sev.lower()}"
            actions = " ".join(f'<span style="color:#00aaff;">• {a}</span>'
                               for a in alert.recommended_actions[:3])
            contrib = " ".join(
                f'<span style="color: rgba(255,255,255,0.55); font-size:0.75rem; font-family: JetBrains Mono;">'
                f'{f}: {v:.3f}</span>' for f, v in alert.contributing_features[:4])
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

    # ── live flows table + export ──
    st.markdown("### 🧾 Live Flows")
    if flows is not None and not flows.empty:
        show_cols = ["Src IP", "Dst IP", "Src Port", "Dst Port", "Protocol",
                     "total_packets", "total_bytes", "SYN_Flag_Cnt", "RST_Flag_Cnt"]
        show_cols = [c for c in show_cols if c in flows.columns]
        st.dataframe(flows[show_cols].tail(200), use_container_width=True, height=280)
        st.download_button(
            "⬇️ Export flows to CSV (for re-training)",
            flows.to_csv(index=False).encode("utf-8"),
            file_name="netseer_live_flows.csv",
            mime="text/csv",
        )
    else:
        st.info("No flows captured yet.")

    st.markdown(
        "<p style='color: rgba(255,255,255,0.35); font-size: 0.75rem;'>"
        "Exported flows use the CIC scheme consumed by <code>scripts/train.py "
        "--data_source csv</code>, so the world model can be fine-tuned on your "
        "own network's traffic for production deployment.</p>", unsafe_allow_html=True)


render_monitor()

# ─── Notes ────────────────────────────────────────────────────────────────────
with st.expander("🧰 Practical roll-out & how this works"):
    st.markdown("""
**Real-world applicability**
- **Live Interface** captures packets on the operator's machine (Scapy + Npcap)
  and analyzes them locally — nothing leaves the network.
- **PCAP Replay** replays any existing capture onto a compressed demo clock, so
  real attack traffic (public threat captures or your own) can be demonstrated
  anywhere, with no capture privileges.
- **Attack Simulator** synthesizes a port scan / brute-force locally and feeds it
  through the exact same pipeline — a self-contained, repeatable demonstration.

**The live path is fully online & deterministic**
- Packets → bidirectional flows (incremental, same as the offline parser).
- Each closed time-window → 76-dim state (log-scale + running-z normalization,
  mapping live statistics onto the World Model's learned signal channels).
- Context of the last N states → K-step forecast, infiltration probability,
  MITRE stage, alerts.

**Closed loop for production**
1. Run the live monitor on the network. 2. Export a session to CSV.
3. `python scripts/train.py --data_source csv --data_dir data/raw`
4. Redeploy — the model now understands *your* traffic.

| Limitation to be honest about | Consequence |
|---|---|
| Pre-trained on CIC-IDS-2018-fidelity states | Cross-network accuracy improves after step 3 |
| Live stages have no ground truth | Alerts are probabilistic hypotheses |
    """)