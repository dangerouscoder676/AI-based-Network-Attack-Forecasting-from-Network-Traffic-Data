"""
NetSeer — one-click console
===========================

The friendly, non-technical entry point to the project. Three buttons:

  🛰️ Scan my Wi-Fi     — discover the network you're on right now, check what's
                         exposed, and ask the AI what an attacker would do next.
  🧪 Simulated attack   — plays a realistic attack on a *fake* network behind the
                         scenes (nothing touches the wire) and shows how the AI
                         spots it, step by step.
  🔭 Live watch         — leave it running while you use the network; a rolling
                         AI alarm if an attack pattern starts.

Technical pages (Dashboard, Analysis, Benchmarks, Network Scout, Live Monitor)
remain available from the sidebar for judges and reviewers.
"""

import sys
import os

# Add project root and the app folder to path. The app folder must precede
# package resolution so `app.py` (this script) does not shadow the `app` package
# while Streamlit has the script directory on sys.path.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
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

setup_page("NetSeer", "Wi-Fi safety check + attack simulation for everyone")

from src.data.network_scanner import (
    COMMON_PORTS,
    arp_discovery,
    discover_hosts,
    guess_local_network,
    host_risk,
    network_cidr_via_scapy,
    network_risk_summary,
    scan_ports,
)
from src.data.live_capture import (
    LiveStateMapper,
    LiveTrafficEngine,
    OnlineScaler,
    extract_window_states,
    list_interfaces,
    STATE_DIM,
)
from src.prediction.attack_simulator import (
    build_phase_plan,
    build_story,
    describe_scan,
    fix_howtos,
    forecast_from_scan,
    quiet_network_forecast,
    SERVICE_DOORS,
    simulate_attack,
)

config = get_config()
CONTEXT_LENGTH = int(config["data"]["context_length"])


# ─── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align: center; padding: 18px 0 6px 0;">
    <h1 style="font-size: 2.6rem; margin-bottom: 6px;">🛡️ NetSeer</h1>
    <p style="color: rgba(255,255,255,0.55); font-size: 1.05rem; font-weight: 300;">
        A gentle look at your Wi-Fi — what could go wrong, and what to do about it.
    </p>
</div>
""", unsafe_allow_html=True)

tab_scan, tab_sim, tab_watch = st.tabs(
    ["🛰️ Scan my Wi-Fi", "🧪 Simulated attack", "🔭 Live watch"])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — Scan my Wi-Fi
# ═════════════════════════════════════════════════════════════════════════════
def render_scan_tab():
    st.markdown(
        "<p style='color: rgba(255,255,255,0.6);'>We look at the network your "
        "computer is on <b>right now</b>, find the devices, check which doors are "
        "open, and ask the AI to predict what an attacker would do next.</p>",
        unsafe_allow_html=True)

    auto_cidr = network_cidr_via_scapy() or guess_local_network()
    with st.expander("⚙️ Advanced (you can usually ignore this)"):
        ifaces = list_interfaces()
        c1, c2 = st.columns([1, 1])
        with c1:
            if ifaces:
                st.selectbox("Network interface", ifaces,
                             help="Used to resolve your computer's address on the link.")
            else:
                st.caption("No capture interfaces found — will fall back to ping-based discovery.")
        with c2:
            st.text_input("Network to check (CIDR)", value=auto_cidr,
                          help="This is your current network, detected automatically. "
                               "Change it only to check another network you own.")
        c3, c4 = st.columns(2)
        with c3:
            method = st.selectbox("Discovery method", ["Auto", "ARP only", "ICMP only"])
        with c4:
            timeout_s = float(st.slider("Probe timeout (s)", 0.2, 3.0, 0.8, step=0.1))
        ports_sel = st.multiselect("Doors to check", sorted(COMMON_PORTS),
                                   default=[22, 80, 443, 445, 3389, 5900, 8080])

    network_key = st.session_state.get("one_scan_network", auto_cidr)
    nt1, nt2 = st.columns([1, 2])
    with nt1:
        st.markdown("#### 🌐 Your network")
    with nt2:
        st.caption(f"Detected as **{network_key}** — press the button to scan it.")

    if st.button("🔍 Scan my network now", type="primary", use_container_width=True):
        bar = st.progress(0, text="Looking for devices…")
        st.session_state["one_progress"] = bar
        try:
            method_key = {"Auto": "auto", "ARP only": "arp", "ICMP only": "icmp"}[method]
            if method_key == "arp":
                hosts = arp_discovery(network_key, timeout=timeout_s + 2,
                                      progress=_progress)
            else:
                hosts = discover_hosts(network_key, prefer=method_key,
                                       timeout=timeout_s + 2, progress=_progress)
            bar.progress(0.5, text=f"{len(hosts)} device(s) found — checking doors…")
            _ports = [int(p) for p in (ports_sel or [])]
            scan_ports(hosts, ports=_ports, timeout_s=timeout_s, progress=_progress)
            bar.progress(0.85, text="Asking the AI what an attacker would do…")
            df = host_risk(hosts)
            import ipaddress as _ip
            total_addressed = len(list(_ip.ip_network(network_key, strict=False).hosts()))
            summary = network_risk_summary(df, total_addressed)

            forecast = {}
            model = load_model("transformer", family="live")
            if model is not None:
                out = forecast_from_scan(df, _ports, model, ctx_win=CONTEXT_LENGTH,
                                         seed=7)
                if "error" not in out:
                    forecast = out
            else:
                forecast = {}

            st.session_state["one_scan_df"] = df
            st.session_state["one_scan_summary"] = summary
            st.session_state["one_scan_forecast"] = forecast
            st.session_state["one_scan_network"] = network_key
            st.session_state["one_scan_when"] = datetime.now().strftime("%H:%M:%S")
            bar.progress(1.0, text="Done")
        except Exception as e:  # noqa: BLE001
            bar.empty()
            st.error(f"Scan failed: {e}")
            st.info("Tip: install **Npcap** (npcap.com) for the richest scan, "
                    "or try the ICMP discovery method.")
        finally:
            st.session_state.pop("one_progress", None)
        st.rerun()

    df = st.session_state.get("one_scan_df")
    summary = st.session_state.get("one_scan_summary")
    forecast = st.session_state.get("one_scan_forecast", {})

    if df is None:
        st.info("Press **Scan my network now** and the results plus the AI's "
                "verdict appear right here.")
        return

    # Plain-language summary
    plain = describe_scan(df, summary, forecast)

    _lvl = plain["level"]
    _lvl_color = {"NONE": "#23d18b", "LOW": "#3fb950", "MEDIUM": "#d29922",
                  "HIGH": "#f85149", "CRITICAL": "#ff3860"}.get(_lvl, "#23d18b")

    t1, t2, t3 = st.columns(3)
    tiles = [
        ("📶", str(summary.get("hosts", 0)), "DEVICES", "#e0e8f0"),
        ("🚪", str(summary.get("open_ports", 0)) or "none", "OPEN DOORS", "#ffb366"),
        ("🛡️", _lvl, plain["level_word"].upper(), _lvl_color),
    ]
    for col, (icon, val, label, color) in zip((t1, t2, t3), tiles):
        with col:
            st.markdown(
                f'<div class="stat-tile">'
                f'<div class="stat-icon">{icon}</div>'
                f'<div class="stat-value" style="color:{color};">{val}</div>'
                f'<div class="stat-label">{label}</div></div>',
                unsafe_allow_html=True)

    verdict = plain["verdict"]
    if verdict["safe"]:
        st.success(f"{verdict['icon']} **{verdict['title']}** — {verdict['text']}")
    else:
        st.warning(f"{verdict['icon']} **{verdict['title']}** — {verdict['text']}")

    # ── safe network: no AI numbers, just reassurance + tips ──
    if plain["safe"]:
        st.markdown(f"**{plain['ai_sentence']}**")
        st.markdown(f"#### {verdict['icon']} Staying safe is easy")
        for rec in plain["recommendations"]:
            st.markdown(f"- {rec}")
        _device_cards(df)
        _fix_section(df)
        st.caption(f"Last scan: **{st.session_state.get('one_scan_when', '')}** · "
                   f"network **{st.session_state.get('one_scan_network', '')}**")
        return

    # ── exposed network: show the AI's read of the attack surface ──
    if forecast.get("peak") is not None:
        st.markdown(f"**{plain['ai_sentence']}**")
        gcol, kcol = st.columns([1, 1.6])
        with gcol:
            st.plotly_chart(create_gauge_chart(forecast["peak"],
                                               title="AI: attack confidence"),
                            use_container_width=True)
        with kcol:
            st.markdown("**What an attacker would do:**")
            chain = " → ".join(dict.fromkeys(forecast.get("seq", [])))
            st.markdown(f"<p style='font-size:1.05rem;'>{chain}</p>",
                        unsafe_allow_html=True)
            _danger = "status-danger" if forecast["peak"] >= 0.6 else (
                "status-warning" if forecast["peak"] >= 0.3 else "status-safe")
            word = ("LIKELY ATTACK PATH" if forecast["peak"] >= 0.6
                    else "SUSPICIOUS PATTERN" if forecast["peak"] >= 0.3 else "LOW RISK")
            st.markdown(f'<span class="status-badge {_danger}" style="display:inline-block;">'
                        f'{word} — {forecast["peak"]:.0%}</span>', unsafe_allow_html=True)
        st.plotly_chart(
            create_infiltration_timeline(
                np.asarray(forecast["probabilities"]).tolist(),
                np.asarray(forecast["stages"]).tolist(),
                title="What the AI expects to happen over the next minutes"),
            use_container_width=True)

    st.markdown("**What you should do:**")
    for rec in plain["recommendations"]:
        st.markdown(f"- {rec}")

    _device_cards(df)
    _fix_section(df)
    st.caption(f"Last scan: **{st.session_state.get('one_scan_when', '')}** · "
               f"network **{st.session_state.get('one_scan_network', '')}**")


def _door_emoji(port: int) -> str:
    if port in SERVICE_DOORS:
        return SERVICE_DOORS[port]["emoji"]
    return "⚙️"


def _device_cards(df):
    """Visual device inventory: IP card + door chips, no spreadsheet feel."""
    st.markdown("#### 🖥️ The devices we found")
    rows = df.sort_values("risk_score", ascending=False)
    cols = st.columns(3)
    for i, (_, h) in enumerate(rows.iterrows()):
        with cols[i % 3]:
            risk = str(h.get("risk_label", "LOW"))
            cls = {"HIGH": "danger", "CRITICAL": "danger", "MEDIUM": "warn"}.get(risk, "")
            doors = str(h.get("services", "") or "")
            chips = ""
            if doors:
                for p in doors.split(","):
                    p = p.strip()
                    if p.isdigit():
                        chips += (f'<span class="door-chip">{_door_emoji(int(p))} {p}</span>')
            else:
                chips = '<span class="door-chip locked">🔒 locked</span>'
            vendor = (str(h.get("vendor", "") or "") or
                      str(h.get("os_guess", "") or "") or "unknown device")
            st.markdown(
                f'<div class="device-card {cls}">'
                f'<div style="display:flex; justify-content:space-between; align-items:center;">'
                f'<b>{h["ip"]}</b>'
                f'<span style="color:{"#" + str(h.get("risk_color", "#3fb950")).lstrip("#")};'
                f' font-weight:700;">{risk}</span></div>'
                f'<div style="color:rgba(255,255,255,0.55); font-size:0.8rem;">{vendor}</div>'
                f'<div style="margin-top:6px;">{chips}</div></div>',
                unsafe_allow_html=True)


def _fix_section(df):
    """Per-door close-guides + a safe 'see it with doors shut' simulation."""
    howtos = fix_howtos(df)
    if not howtos["devices"]:
        return

    st.markdown("#### 🚪 Your open doors — and how to close each one")
    seen = {}
    for dev in howtos["devices"]:
        for door in dev["doors"]:
            seen.setdefault(door["port"], door)

    cols = st.columns(2)
    for i, item in enumerate(seen.values()):
        with cols[i % 2]:
            st.markdown(
                f'<div class="howto-card">'
                f'<div style="font-weight:700;">{item["emoji"]} Port {item["port"]} — {item["name"]}</div>'
                f'<p style="color:rgba(255,255,255,0.6); font-size:0.85rem; margin:6px 0;">'
                f'{item["why"]}</p>'
                f'<p style="font-size:0.85rem; margin:6px 0;"><b>How to close it:</b> {item["how"]}</p>'
                f'</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        "**See what changes before you touch anything?** This re-runs the AI on a "
        "version of your network with those doors shut — it does **not** change "
        "anything on your devices.")
    if st.button("🧪 Show me this network with those doors closed", type="secondary"):
        model = load_model("transformer", family="live")
        if model is None:
            st.error("The AI model isn't here yet. Run `python scripts/train_live.py` once.")
        else:
            with st.spinner("Simulating a tightened network…"):
                st.session_state["one_fix"] = quiet_network_forecast(model)
            st.rerun()

    fix = st.session_state.get("one_fix")
    if fix and "peak" in fix:
        g_col, t_col = st.columns([1, 2])
        with g_col:
            st.plotly_chart(create_gauge_chart(fix["peak"], title="AI read once doors are shut"),
                            use_container_width=True)
        with t_col:
            quiet = fix["peak"] < 0.3
            st.markdown(
                f"**With nothing open to grab — and nobody probing — the AI reads the "
                f"network as {'quiet' if quiet else 'noticeably calmer'} "
                f"({fix['peak']:.0%} peak).**", unsafe_allow_html=True)
            st.markdown(
                "That's the difference between doors open and doors shut. Follow the "
                "steps above, then press **Scan my network now** again to see it live.")


def _progress(fraction):
    if "one_progress" in st.session_state:
        st.session_state["one_progress"].progress(min(fraction, 1.0))


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — Simulated attack on a fake network
# ═════════════════════════════════════════════════════════════════════════════
def render_sim_tab():
    st.markdown(
        "<p style='color: rgba(255,255,255,0.6);'>A make-believe network (computers, "
        "a server, a printer, some phones) is attacked in the background — "
        "<b>nothing real is touched, no packets leave your machine</b>. Watch how "
        "the attack would play out and how the AI tracks it.</p>", unsafe_allow_html=True)

    c1, c2 = st.columns([1, 2])
    with c1:
        seed = st.slider("Different story each time", 0, 99, 7,
                         help="Pick another number for a slightly different attack.")
        play = st.button("▶️ Play the simulated attack", type="primary",
                         use_container_width=True)
    with c2:
        st.markdown(
            "<div class='glass-card' style='padding:14px; margin-top:4px;'>"
            "<div style='font-size:0.8rem; color:rgba(255,255,255,0.5);'>The fake "
            "network behind the curtain</div>"
            "<div style='font-size:0.95rem;'>🏢 A small business: a file server, an "
            "admin laptop, a few phones and a printer.<br>"
            "🕵️ A stranger on the Wi-Fi tries to get in.<br>"
            "🧠 NetSeer watches every moment and calls it out.</div></div>",
            unsafe_allow_html=True)

    if play:
        model = load_model("transformer", family="live")
        if model is None:
            st.error("The AI model isn't here yet. Run `python scripts/train_live.py` once.")
            return
        phases = build_phase_plan(n_benign=20, attacks=[(1, 5), (2, 4), (3, 4)])
        with st.spinner("Running the attack behind the scenes…"):
            st.session_state["one_sim"] = simulate_attack(model, phases, seed=int(seed))
        st.rerun()

    sim = st.session_state.get("one_sim")
    if not sim:
        st.info("Press **Play the simulated attack** — it takes a few seconds.")
        return
    if "error" in sim:
        st.error(sim["error"])
        return

    st.markdown("#### 📖 What just happened (in plain words)")
    for line in sim["story"]:
        st.markdown(f"- {line}")

    probs = np.array([t["prob"] for t in sim["timeline"]])
    stages = np.array([t.get("stage", 0) for t in sim["timeline"]])

    gcol, tcol = st.columns([1, 1.7])
    with gcol:
        peak = sim["peak"]
        st.markdown(
            f'<div class="glass-card" style="text-align:center; padding:10px;">'
            f'<div style="font-size:0.8rem; color:rgba(255,255,255,0.5);">AI caught it: '
            f'{peak:.0%} confidence</div></div>', unsafe_allow_html=True)
        st.plotly_chart(create_gauge_chart(peak, title="Attack detection"),
                        use_container_width=True)
    with tcol:
        st.plotly_chart(
            create_infiltration_timeline(probs.tolist(), stages.tolist(),
                                         title="Trusted traffic (left) → attack detected (right)"),
            use_container_width=True)

    st.markdown(
        "<p style='font-size:0.8rem; color:rgba(255,255,255,0.35);'>"
        "The blue-marked moments are where the model first grew suspicious. In a "
        "real network this is the alarm going off one or two windows before the "
        "damage is done.</p>", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — Live watch (protect your network)
# ═════════════════════════════════════════════════════════════════════════════
def render_watch_tab():
    st.markdown(
        "<p style='color: rgba(255,255,255,0.6);'>Leave this running while you use "
        "your Wi-Fi. If the AI spots the start of an attack pattern, it raises an "
        "alarm here.</p>", unsafe_allow_html=True)

    running = st.session_state.get("live_engine") is not None and st.session_state.get("live_engine").running

    if not running:
        c1, c2 = st.columns([1, 2])
        with c1:
            ifaces = list_interfaces()
            interface = st.selectbox("Interface", ifaces or ["(capture unavailable)"],
                                     index=0 if ifaces else None)
        with c2:
            threshold = st.slider("Alert when the AI is this suspicious", 0.2, 0.9, 0.6, 0.05)
            window_size = st.slider("Check every (seconds)", 2, 20, 5)
        can_start = bool(ifaces) and not interface.startswith("(")
        if st.button("▶️ Start watching", type="primary", use_container_width=True,
                     disabled=not can_start):
            engine = LiveTrafficEngine()
            engine.start_interface(ifaces[0] if (not interface or interface.startswith("(")) else interface)
            st.session_state["live_engine"] = engine
            st.session_state["live_runtime"] = {
                "mapper": LiveStateMapper(), "scaler": OnlineScaler(STATE_DIM),
                "states": [], "next_window": 0, "history": [], "last": None,
                "window_size": float(window_size), "k_steps": 8,
                "threshold": float(threshold), "context_length": CONTEXT_LENGTH,
            }
            st.rerun()
    else:
        st.info(f"⏺ Watching **{st.session_state['live_engine'].source}** — "
                "the view below refreshes by itself.")
        if st.button("🛑 Stop watching", type="secondary"):
            st.session_state["live_engine"].stop()
            st.rerun()

    @st.fragment(run_every=(2.0 if running else None))
    def watch_fragment():
        engine = st.session_state.get("live_engine")
        rt = st.session_state.get("live_runtime")
        if engine is None or rt is None:
            st.markdown("<p style='color:rgba(255,255,255,0.4);'>Press "
                        "**Start watching**.</p>", unsafe_allow_html=True)
            return
        if not engine.running:
            status = engine.status()
            if status.get("error"):
                st.error(f"Watching stopped: {status['error']}")
            else:
                st.warning(f"Watching ended — {status['packets']:,} packets seen.")
            return

        model = load_model("transformer", family="live")
        if model is None:
            st.error("The AI model isn't here yet. Run `python scripts/train_live.py` once.")
            return

        status = engine.status()
        flows = engine.snapshot_flows()
        if flows is not None and not flows.empty and status.get("sim_t0"):
            new_states, rt["next_window"] = extract_window_states(
                flows, float(status["sim_t0"]), rt["window_size"],
                rt["mapper"], rt["scaler"], rt["next_window"])
            for vec in new_states:
                rt["states"].append(vec)
                if len(rt["states"]) > 512:
                    rt["states"] = rt["states"][-512:]

        prob_now = None
        if len(rt["states"]) >= rt["context_length"]:
            ctx = np.asarray(rt["states"][-rt["context_length"]:], dtype=np.float32)
            rollout = run_rollout(model, ctx, rt["k_steps"])
            if rollout is not None:
                prob_now = float(np.asarray(rollout["probabilities"])[0])
                rt["history"].append(prob_now)
                if len(rt["history"]) > 120:
                    rt["history"] = rt["history"][-120:]

        thresh = rt.get("threshold", 0.6)
        if prob_now is None:
            st.info("Gathering a few seconds of traffic before judging…")
            return

        word = (f"! ALERT — an attack pattern is starting"
                if prob_now >= thresh
                else "pay attention" if prob_now >= 0.35
                else "what we're seeing looks normal")
        if prob_now >= thresh:
            st.error(f"🚨 **{word}** (AI confidence {prob_now:.0%}). "
                     "Stop unfamiliar activity, disconnect the odd device, and run "
                     "**Scan my Wi-Fi** above.")
        elif prob_now >= 0.35:
            st.warning(f"⚠️ **{word}** — AI confidence {prob_now:.0%}. "
                       "Keep watching.")
        else:
            st.success(f"✅ {word} — AI confidence {prob_now:.0%}.")

        gcol, tcol = st.columns([1, 1.7])
        with gcol:
            st.plotly_chart(create_gauge_chart(prob_now, title="AI: right now"),
                            use_container_width=True)
        with tcol:
            if len(rt["history"]) >= 2:
                st.plotly_chart(
                    create_infiltration_timeline(rt["history"][-60:],
                                                 [0] * len(rt["history"][-60:]),
                                                 title="AI confidence over the last minutes"),
                    use_container_width=True)
            else:
                st.markdown("<p style='color:rgba(255,255,255,0.4);'>Confidence "
                            "history will appear here.</p>", unsafe_allow_html=True)

    watch_fragment()


with tab_scan:
    render_scan_tab()
with tab_sim:
    render_sim_tab()
with tab_watch:
    render_watch_tab()

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:rgba(255,255,255,0.35); font-size:0.8rem;'>"
    "Want the whole story? The sidebar has the full dashboard, benchmarks and "
    "deep-dive analysis pages.<br>NetSeer — AI network attack forecasting.</p>",
    unsafe_allow_html=True)