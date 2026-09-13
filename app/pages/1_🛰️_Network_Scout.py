"""
NetSeer — Network Scout (active reconnaissance)
===============================================
Points the system at a **real, live network**: discovers hosts, runs a light
TCP connect scan on common ports, scores the attack surface with an
interpretable heuristic mapped to MITRE ATT&CK, and then feeds the scanner's
own probe traffic through the World Model so the forecast reacts to the
reconnaissance it just performed.

This is the "connect to a network, analyse it, predict threats" capability for
live demos (own hotspot / lab / wired segment). Requirements: at least the
machine's interface reachable on the target subnet; Npcap optional (used only
for ARP host enumeration — ICMP `ping` is the zero-install fallback).
"""

import sys
import os

_PAGES_DIR = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.dirname(_PAGES_DIR)
_PROJECT_ROOT = os.path.dirname(_APP_DIR)
for _p in (_APP_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from theme import setup_page
from session import get_config, load_model, run_rollout, STAGE_NAMES
from components.charts import create_gauge_chart, create_infiltration_timeline

setup_page("NetSeer", "Network Scout — discover, assess, forecast")

from src.data.network_scanner import (
    COMMON_PORTS,
    arp_discovery,
    discover_hosts,
    emit_probe_flows,
    guess_local_network,
    host_risk,
    icmp_discovery,
    network_cidr_via_scapy,
    network_risk_summary,
    scan_ports,
    tcp_connect_scan_one,
)
from src.data.live_capture import (
    LiveStateMapper,
    OnlineScaler,
    extract_window_states,
    STATE_DIM,
    list_interfaces,
)

config = get_config()
CONTEXT_LENGTH = int(config["data"]["context_length"])
STATE_DIM = int(config["features"]["num_state_features"])

st.markdown("### 🛰️ Network Scout")
st.markdown(
    "<p style='color: rgba(255,255,255,0.6); font-size: 0.9rem;'>"
    "Interrogates the network you connect the laptop to: finds live hosts, "
    "checks which common services are exposed, maps the exposure to MITRE "
    "ATT&amp;CK and feeds the findings through the World Model to forecast the "
    "attacker's next move.</p>",
    unsafe_allow_html=True,
)


# ─── Target configuration ─────────────────────────────────────────────────────
st.markdown("### 🎯 Scan Target")

auto_cidr = network_cidr_via_scapy() or guess_local_network()
with st.expander("Interface & target network", expanded=True):
    ifaces = list_interfaces()
    c1, c2 = st.columns([1, 1])
    with c1:
        if ifaces:
            st.selectbox("Network interface", ifaces, help="Used to resolve the local address on the link.")
        else:
            st.caption("No capture interfaces enumerated — Npcap absent, falling back to ICMP discovery.")
    with c2:
        auto_label = "Network (CIDR) — auto: " + auto_cidr
        network = st.text_input("Network (CIDR)", value=auto_cidr,
                                help="e.g. 192.168.1.0/24. Scan only on networks you own.")
    c3, c4, c5 = st.columns(3)
    method = c3.selectbox("Host discovery", ["Auto (ARP→ICMP)", "ARP only", "ICMP only"],
                          help="ARP needs Npcap and returns MACs; ICMP needs no install.")
    ports_sel = c4.multiselect("Ports to probe", sorted(COMMON_PORTS),
                               default=[22, 80, 443, 445, 3389, 5900, 8080],
                               help="Curated common ports — a light, polite scan.")
    timeout_s = float(c5.slider("Probe timeout (s)", 0.2, 3.0, 0.8, step=0.1))

scan_ready = True
try:
    import ipaddress as _ip
    scan_net = _ip.ip_network(network, strict=False)
except ValueError:
    st.error(f"Invalid CIDR: {network}")
    scan_ready = False

_ports = [int(p) for p in (ports_sel or [])]


def _run_progress(fraction):
    if "scout_progress" in st.session_state:
        st.session_state["scout_progress"].progress(min(fraction, 1.0))


# ─── Run scan ─────────────────────────────────────────────────────────────────
col_b, col_x = st.columns([1, 2])
with col_b:
    run_scan = st.button("🔍 Scan Network", type="primary", disabled=not scan_ready,
                         help="Sends real discovery + TCP connect probes on the target network.")
with col_x:
    if st.session_state.get("scout_df") is not None and not st.session_state.get("scout_df").empty:
        st.caption(f"Last scan finished **{st.session_state['scout_when']}** — "
                   f"**{st.session_state['scout_network']}**")

if run_scan:
    bar = st.progress(0, text="Starting…")
    st.session_state["scout_progress"] = bar
    try:
        method_key = {"Auto (ARP→ICMP)": "auto", "ARP only": "arp", "ICMP only": "icmp"}[method]
        if method_key == "arp":
            hosts = arp_discovery(network, timeout=timeout_s + 2, progress=_run_progress)
        else:
            hosts = discover_hosts(network, prefer=method_key, timeout=timeout_s + 2,
                                   progress=_run_progress)
        bar.progress(0.45, text=f"Discovery done — {len(hosts)} host(s) live")
        if not hosts:
            st.warning("No live hosts discovered. Check the network CIDR / your link, "
                       "or switch discovery method.")
        scan_ports(hosts, ports=_ports, timeout_s=timeout_s, progress=_run_progress)
        bar.progress(0.85, text="Scoring attack surface…")
        df = host_risk(hosts)
        total_addressed = len(list(scan_net.hosts())) if scan_ready else 254
        summary = network_risk_summary(df, total_addressed)
        st.session_state["scout_df"] = df
        st.session_state["scout_summary"] = summary
        st.session_state["scout_network"] = network
        st.session_state["scout_when"] = datetime.now().strftime("%H:%M:%S")
        bar.progress(1.0, text="Scan complete")
    except Exception as e:  # noqa: BLE001
        bar.empty()
        st.error(f"Scan failed: {e}")
        st.info("Install **Npcap** (https://npcap.com) for ARP + raw capture, or use the ICMP-only method.")
    finally:
        st.session_state.pop("scout_progress", None)
    st.rerun()


# ─── Results ──────────────────────────────────────────────────────────────────
df = st.session_state.get("scout_df")
summary = st.session_state.get("scout_summary")

if df is None:
    st.markdown(
        "<p style='color: rgba(255,255,255,0.4);'>Run a scan to see the live inventory, "
        "attack-surface score and World Model forecast.</p>", unsafe_allow_html=True)
    st.stop()

# header metrics
_lvl = summary.get("level", "NONE")
_lvl_color = {"NONE": "#23d18b", "LOW": "#3fb950", "MEDIUM": "#d29922",
              "HIGH": "#f85149", "CRITICAL": "#ff3860"}.get(_lvl, "#23d18b")
st.markdown(
    f'<div class="glass-card"><div style="display:flex; justify-content:space-between;'
    f'align-items:center;">'
    f'<span style="font-size:1.5rem;">🌐 {st.session_state["scout_network"]}</span>'
    f'<span style="font-size:1.6rem; color:{_lvl_color}; font-weight:800;">'
    f'ATTACK SURFACE: {_lvl}</span></div>'
    f'<div style="display:flex; gap:32px; margin-top:12px; color:rgba(255,255,255,0.7);">'
    f'<span>Hosts: <b>{summary["hosts"]}</b></span>'
    f'<span>Open ports: <b>{summary["open_ports"]}</b></span>'
    f'<span>Exposed hosts: <b>{summary["exposed_hosts"]}</b></span>'
    f'<span>High/CRITICAL: <b>{summary["risk_high"]}</b></span></div></div>', unsafe_allow_html=True)

if summary.get("mitre"):
    st.markdown("**MITRE ATT&CK techniques this surface invites:**  " +
                "  •  ".join(f"`{m}`" for m in summary["mitre"]))

st.markdown("### 🖥️ Host Inventory")


def _style_risk_row(row):
    if row["risk_label"] in ("HIGH", "CRITICAL"):
        return [f"color:#f85149; font-weight:700;"] * len(row)
    return [""] * len(row)


_show = df[["ip", "mac", "vendor", "os_guess", "open_ports", "services",
            "risk_label", "likely_target", "mitre_techniques"]].copy()
st.dataframe(_show.style.apply(_style_risk_row, axis=1),
             use_container_width=True, height=min(320, 40 * (len(df) + 1)))

# per-host cards
st.markdown("### 🎯 Likely attacker progression")
_hosts_by_risk = df.sort_values("risk_score", ascending=False)
cols_per_row = 3
for i in range(0, len(_hosts_by_risk), cols_per_row):
    row_cols = st.columns(cols_per_row)
    for j in range(cols_per_row):
        if i + j >= len(_hosts_by_risk):
            break
        h = _hosts_by_risk.iloc[i + j]
        with row_cols[j]:
            _rc = h["risk_label"]
            _cc = h["risk_color"].replace("#", "")
            st.markdown(f"""
            <div class="glass-card" style="border-left: 3px solid #{_cc};">
                <div style="display:flex; justify-content:space-between;">
                    <b>{h['ip']}</b>
                    <span style="color:#{_cc}; font-weight:700;">{_rc}</span>
                </div>
                <div style="color:rgba(255,255,255,0.55); font-size:0.8rem;">
                    {h['vendor'] or h['os_guess'] or 'unknown device'}
                </div>
                <div style="font-size:0.8rem;">
                    <span>Open: </span>{" ".join(f"`{p}`" for p in h['services'].split(',') if p) or "none"}
                </div>
            </div>""", unsafe_allow_html=True)

# ─── World Model forecast from the scan ───────────────────────────────────────
st.markdown("---")
st.markdown("### 🧠 World Model Forecast on the scan")
model = load_model("transformer", family="live")
if model is None:
    st.error("No live-trained World Model weights found. Run `python scripts/train_live.py` first.")
    st.stop()

with st.expander("How the forecast is built", expanded=False):
    st.markdown(
        "The scanner's **own** SYN/connect probes are routed through the same "
        "flow → state → world-model pipeline used by the Live Monitor (nothing is "
        "sent twice on the wire). The model sees the reconnaissance signature — "
        "high SYN ratio, rising destination-port diversity — and predicts how the "
        "attack would progress in the next K windows.")

w1, w2, w3 = st.columns([1, 1, 1])
with w1:
    ctx_win = st.number_input("Context (benign windows before the scan)", 6, 20, 12, step=1,
                              help="Length of the benign baseline the online scaler learns from.")
with w2:
    attack_win = st.number_input("Recon persistence (windows)", 1, 5, 1, step=1,
                                 help="How many consecutive scan-pattern windows the model sees. "
                                      "One burst is the cleanest demonstration: the world model "
                                      "recognises the scan as reconnaissance on its own.")
with w3:
    k_steps = st.slider("Forecast horizon (steps)", 3, 20, 10)

_forecast = None
_ctx_state = None
if st.button("🔮 Forecast infiltration from scan findings", type="secondary"):
    with st.spinner("Running K-step forward simulation…"):
        from scapy.all import IP, TCP, UDP
        from src.data.pcap_parser import FlowAggregator

        agg = FlowAggregator()
        try:
            from scapy.all import get_if_addr
            src_ip = get_if_addr() or "10.10.10.10"
        except Exception:  # noqa: BLE001
            src_ip = "10.10.10.10"

        win_s = 5.0
        base = 1_700_000_000.0

        emit_probe_flows(df, agg, src_ip=src_ip, benign_before=int(ctx_win),
                         seed=int(np.random.randint(0, 2 ** 31)),
                         window_size_s=win_s, t0=base,
                         attack_windows=int(attack_win),
                         probed_ports=_ports or list(COMMON_PORTS))

        flows = agg.snapshot()
        mapper = LiveStateMapper()
        scaler = OnlineScaler(STATE_DIM)
        states, _nxt = extract_window_states(flows, base, win_s, mapper, scaler, 0)
        if len(states) < 2:
            st.warning("Not enough flow detail to form a state sequence. "
                       "Re-run the scan or use a lighter port list.")
        else:
            ctx = np.asarray(states[-CONTEXT_LENGTH:], dtype=np.float32)
            rollout = run_rollout(model, ctx, int(k_steps))
            if rollout:
                _forecast = {
                    "probabilities": np.asarray(rollout["probabilities"]),
                    "stages": np.asarray(rollout["stages"]),
                    "peak": float(np.asarray(rollout["probabilities"]).max()),
                    "seq": [STAGE_NAMES[int(s)] for s in rollout["stages"]],
                }
                _ctx_state = ctx[-1]
                st.session_state["scout_forecast"] = _forecast
            else:
                st.error("Forecast failed — model returned no rollout.")

_forecast = st.session_state.get("scout_forecast")

if _forecast is None:
    st.caption("Run the forecast to see the shader below.")
else:
    peak = _forecast["peak"]
    pred_stage = _forecast["stages"][int(np.argmax(_forecast["probabilities"]))]
    gcol, kcol = st.columns([1, 1.6])
    with gcol:
        st.plotly_chart(create_gauge_chart(peak, title="Predicted Infiltration"), use_container_width=True)
    with kcol:
        st.markdown(f"#### 🔗 Predicted ATT&CK progression")
        st.markdown(" → ".join(f"`{s}`" for s in dict.fromkeys(_forecast["seq"])))
        _danger = "status-danger" if peak >= 0.6 else ("status-warning" if peak >= 0.3 else "status-safe")
        st.markdown(f'<span class="status-badge {_danger} threat-active">'
                    f'{"INFILTRATION LIKELY" if peak >= 0.5 else "RECON PATTERN" if peak >= 0.3 else "LOW RISK"}'
                    f' — peak {peak:.0%}</span>', unsafe_allow_html=True)
        st.markdown(
            f"<p style='color:rgba(255,255,255,0.6); font-size:0.85rem;'>"
            f"The exposed hosts the scan found would make the first step in this "
            f"chain (<b>{STAGE_NAMES[int(pred_stage)]}</b>) the attacker's obvious "
            f"entry point.</p>", unsafe_allow_html=True)
    st.plotly_chart(
        create_infiltration_timeline(_forecast["probabilities"].tolist(),
                                     _forecast["stages"].tolist(),
                                     title=f"K-step forecast from scan (K={k_steps})"),
        use_container_width=True,
    )

# ─── Export + notes ───────────────────────────────────────────────────────────
c1, c2 = st.columns(2)
with c1:
    st.download_button("⬇️ Export inventory CSV", df.to_csv(index=False).encode("utf-8"),
                       file_name=f"netseer_scan_{st.session_state['scout_network'].replace('/','_')}.csv",
                       mime="text/csv")
with c2:
    st.caption("Discovery: ARP (MAC+Vendor) / ICMP (TTL OS-typing) · Port test: TCP connect "
               "· Scoring: interpretive exposure heuristic mapped to MITRE ATT&CK "
               "(T1595/T1046 reconnaissance, T1190 initial access).")

with st.expander("🧰 Practical notes for the judges' demo"):
    st.markdown("""
**Setting up the demo**
1. Connect this laptop (and a couple of phones/laptops) to your own hotspot or lab router.
2. Optionally install **Npcap** once for ARP discovery + live capture (`pip install scapy==2.6.1`).
3. Open this page → the network CIDR auto-detects → **Scan Network**.
4. Then repeat the scan while watching the **Live Monitor** (or the forecast here) — the
   World Model flags the recon burst and predicts the next stage of the kill chain.

**Honest limitations**
- The port list is curated and light by design — it is discovery, not a full pentest.
- TCP connect scan completes handshakes; some hosts log failed connections.
- Predictions are probabilistic hypotheses on top of a live-trained model that was
  validated on synthetic traffic — treat the probability as a risk cue, not ground truth.
- Windows Firewall may prompt to allow Python's outbound connects on first scan.
""")