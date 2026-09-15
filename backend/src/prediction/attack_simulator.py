"""
Attack Simulator + plain-language reporting for the layman console.

The simulator builds a *fake* network, plays realistic attack traffic against it
behind the scenes (never on the wire), and asks the trained World Model to score
each moment. The reporting helpers translate the numeric results into sentences a
non-technical user understands.

Nothing here touches a real network — all traffic is synthesised into flow
aggregators, exactly like ``scripts/train_live.py`` does for training data.
"""

from typing import Dict, List
from collections import Counter

import numpy as np

from src.data.live_capture import (
    LiveStateMapper,
    OnlineScaler,
    STATE_DIM,
    extract_window_states,
)

try:
    from scripts.train_live import _GENERATORS
except Exception:  # noqa: BLE001  (import-time guard; fall through at runtime)
    _GENERATORS = None

STAGE_NAMES = [
    "Benign",
    "Reconnaissance",
    "Initial Access",
    "Lateral Movement",
    "Command & Control",
    "Exfiltration",
]

# Plain-word equivalents for the stages the model knows.
_STAGE_PLAIN = {
    "Benign": "normal",
    "Reconnaissance": "reconnaissance (an attacker is looking for weak spots)",
    "Initial Access": "breaking in through an open service",
    "Lateral Movement": "moving between your devices",
    "Command & Control": "taking control of a device",
    "Exfiltration": "stealing your data out",
}

# What the injected simulation phases look like to a layman.
_PHASE_PLAIN = {
    1: "an attacker scans your network for open doors (reconnaissance)",
    2: "the attacker hammers a login page to brute-force a password",
    3: "the attacker floods a server to knock it offline",
}


def build_phase_plan(n_benign: int = 24, attacks: List[tuple] = None) -> List[int]:
    """A phase plan: ``n_benign`` quiet windows, then the given attack phases.

    ``attacks`` is a list of ``(phase, length)`` runs, e.g. ``[(1, 4), (2, 4)]``.
    Returns a list of per-window phases (0 = benign).
    """
    plan = [0] * int(n_benign)
    for phase, length in (attacks or [(1, 4), (2, 4)]):
        plan += [int(phase)] * int(length)
    plan += [0] * 4  # calm tail so the AI can "recover"
    return plan


def simulate_attack(
    model,
    phases: List[int],
    seed: int = 7,
    context_length: int = 16,
    window_size_s: float = 5.0,
    k_steps: int = 10,
    flow_transform=None,
) -> dict:
    """Replay the phase plan through the live pipeline and score every window.

    Returns a dict with ``timeline`` (list of per-window dicts: window index,
    injected phase + plain name, model's predicted stage + plain name, next-step
    infiltration probability), ``peak`` probability, the first predicted stage
    family, and plain-language ``story`` sentences.

    ``flow_transform`` is an optional callable ``df -> df`` applied to each
    window's aggregated flows *before* they are mapped to a model state. This is
    how the what-if simulator models a counterfactual defence (e.g. dropping the
    attacker's flows to a blocked port).
    """
    if _GENERATORS is None:
        return {"error": "Could not load traffic generators (scripts/train_live.py)."}
    import scripts.train_live as train_live
    rng = np.random.RandomState(seed)
    train_live._RNG = rng  # the generators also consult the module-level RNG
    mapper = LiveStateMapper()
    scaler = OnlineScaler(STATE_DIM)
    states: List[np.ndarray] = []
    timeline = []
    t0 = 1_700_000_000.0
    peak = 0.0
    stage_name_peak = "Benign"

    for w, phase in enumerate(phases):
        agg = _make_aggregator()
        _GENERATORS[phase](agg, t0 + w * window_size_s, rng)
        flows = agg.snapshot()
        if flow_transform is not None:
            flows = flow_transform(flows)
        if flows is None or flows.empty:
            timeline.append({"window": w, "phase": int(phase), "prob": 0.0, "stage": 0,
                             "phase_name": _PHASE_PLAIN.get(int(phase), "normal")})
            continue
        states.append(mapper.state(flows, scaler, window_size_s=window_size_s))

        prob = 0.0
        stage = 0
        if len(states) >= context_length:
            ctx = np.asarray(states[-context_length:], dtype=np.float32)
            rollout = _rollout(model, ctx, k_steps)
            if rollout is not None:
                p = rollout["probabilities"]
                s = rollout["stages"]
                if len(p):
                    prob = float(p[0])
                    stage = int(s[0])
        if prob > peak:
            peak = prob
            stage_name_peak = STAGE_NAMES[stage] if 0 <= stage < len(STAGE_NAMES) else "unknown"
        timeline.append({
            "window": w,
            "phase": int(phase),
            "phase_name": _PHASE_PLAIN.get(int(phase), "normal"),
            "prob": prob,
            "stage": stage,
            "stage_name": STAGE_NAMES[stage] if 0 <= stage < len(STAGE_NAMES) else "unknown",
        })

    story = build_story(timeline, peak, stage_name_peak)
    return {
        "timeline": timeline,
        "peak": peak,
        "stage_name_peak": stage_name_peak,
        "story": story,
    }


def build_story(timeline: List[dict], peak: float, stage_name_peak: str) -> List[str]:
    """Plain-word narration of what just played out."""
    plain = _STAGE_PLAIN.get(stage_name_peak, "under attack")
    lines = []
    phases_seen = []
    for t in timeline:
        pn = t.get("phase_name")
        if pn not in phases_seen:
            phases_seen.append(pn)
    lines.append(
        "The simulation starts quietly — then " +
        (" and ".join(phases_seen) if phases_seen else "the attack begins") +
        "."
    )
    if peak >= 0.6:
        lines.append(
            f"The AI detected this immediately. At its peak the model was "
            f"{peak:.0%} sure an intrusion was underway — pointing at "
            f"{stage_name_peak.lower()} ({plain})."
        )
    else:
        lines.append(
            f"The AI kept the threat low ({peak:.0%} at peak) — "
            "the pattern looked normal to it."
        )
    lines.append(
        "In a real network this is the alerting moment: NetSeer would raise an "
        "alarm one or more windows BEFORE the damage completes."
    )
    return lines


def describe_scan(df, summary: dict, forecast: dict) -> dict:
    """Turn scan numbers into sentences a layman can act on."""
    n_hosts = int(summary["hosts"])
    exposed = int(summary["exposed_hosts"])
    open_total = int(summary["open_ports"])
    level = summary["level"]

    level_word = {"NONE": "no obvious weak spots",
                  "LOW": "a little exposed",
                  "MEDIUM": "moderately exposed",
                  "HIGH": "very exposed"}.get(level, "exposed")

    headline = (f"Found **{n_hosts}** device(s) on your network. "
                f"**{exposed}** of them have open doors that a stranger could try.")

    safe = level in ("NONE", "LOW")
    if safe:
        ai_sentence = "No urgent problems were found — the tips below help you keep it that way."
    else:
        ai_sentence = (
            "Run the AI forecast to see what an attacker would do next."
            if not forecast
            else f"The AI's verdict: an attacker's first move here would be "
                 f"**{forecast['stage_display']}**, with up to **{forecast['peak']:.0%}** "
                 f"confidence that this leads to an intrusion."
        )

    if safe:
        verdict = {
            "safe": True,
            "icon": "✅",
            "title": "Your Wi-Fi looks safe right now",
            "text": (f"{n_hosts} device(s), {open_total} open door(s) — a stranger "
                     f"has little to work with. Keep the Wi-Fi password strong and "
                     f"you're in good shape."),
        }
    else:
        verdict = {
            "safe": False,
            "icon": "⚠️",
            "title": "Your Wi-Fi has weak spots worth tightening",
            "text": (f"{exposed} of {n_hosts} device(s) have open doors. "
                     f"The forecast below shows what a stranger could try."),
        }

    recommendations = []
    if open_total:
        recommendations.append("Close any network doors you don't use every day.")
    if level in ("HIGH", "MEDIUM"):
        recommendations.append("Check your router / hotspot's admin settings and change the default password.")
    if exposed:
        recommendations.append("Make sure the devices with open doors have up-to-date software.")
        if summary.get("mitre"):
            tech = " · ".join(str(m).split(" ", 1)[0] for m in summary["mitre"])
            recommendations.append(f"The exposed patterns map to common attack techniques: {tech}.")
    if not recommendations:
        recommendations.append("Nothing urgent found. Keep this Wi-Fi password-strong and you're in good shape.")

    return {
        "headline": headline,
        "level": level,
        "level_word": level_word,
        "safe": safe,
        "verdict": verdict,
        "ai_sentence": ai_sentence,
        "recommendations": recommendations,
    }


# ─── Service close-guides (what each "open door" is + how to close it) ────────
SERVICE_DOORS = {
    20: {"name": "FTP (file transfer)", "emoji": "📁",
         "why": "Lets a stranger connect and move files to/from a device.",
         "how": "On the device, turn off the file-upload/test service (older IoT and NAS drives have it)."},
    21: {"name": "FTP (file transfer)", "emoji": "📁",
         "why": "Lets a stranger connect and move files to/from a device.",
         "how": "On the device, turn off the file-upload/test service (older IoT and NAS drives have it)."},
    22: {"name": "SSH remote command line", "emoji": "🔑",
         "why": "Gives a stranger a terminal on the device if they guess the password.",
         "how": "On the device: stop the SSH service, or disable password logins and use a key only."},
    23: {"name": "Telnet remote console", "emoji": "🖥️",
         "why": "An old remote console that sends the password in plain text.",
         "how": "Turn off Telnet in the device's settings — always prefer SSH, or nothing."},
    25: {"name": "SMTP mail relay", "emoji": "📨",
         "why": "An old mail service that spammers can abuse to send junk.",
         "how": "Disable the mail service on the device unless you genuinely run a mail server."},
    53: {"name": "DNS helper", "emoji": "🧭",
         "why": "Some routers expose this to help guests; it is usually harmless but unneeded.",
         "how": "Leave it on if this is your router, or disable only if you know it isn't needed."},
    80: {"name": "Web page (HTTP)", "emoji": "🌐",
         "why": "A device is serving a web page anyone on the Wi-Fi can open.",
         "how": "If it's a printer/router admin page, leave it but keep the admin password strong."},
    443: {"name": "Secure web page (HTTPS)", "emoji": "🔐",
         "why": "Usually a normal service (web interface) — low worry, keep software updated.",
         "how": "Usually fine to leave on; just keep the device's software patched."},
    445: {"name": "Windows file sharing (SMB)", "emoji": "🗂️",
         "why": "Old Windows file-sharing — a very common target for attacks.",
         "how": "On Windows: Settings → Apps → Optional features → uninstall SMB 1.x; in File Explorer, "
               "turn off file and printer sharing when not needed."},
    139: {"name": "Old Windows NetBIOS", "emoji": "🗂️",
         "why": "Legacy Windows sharing name-service that attackers scan for.",
         "how": "On Windows: Network settings → turn off file sharing over NetBIOS (SMB 1/NetBIOS)."},
    3389: {"name": "Remote desktop (RDP)", "emoji": "🖥️",
          "why": "Lets a stranger control the computer screen if they get in.",
          "how": "On Windows: Settings → System → Remote Desktop → turn it OFF unless you really need it."},
    5900: {"name": "Remote screen (VNC)", "emoji": "🖥️",
          "why": "A remote-control program that is easy to brute-force.",
          "how": "Quit/disable the VNC server app on that device when you aren't using it."},
    5901: {"name": "Remote screen (VNC)", "emoji": "🖥️",
          "why": "A remote-control program that is easy to brute-force.",
          "how": "Quit/disable the VNC server app on that device when you aren't using it."},
    3306: {"name": "MySQL database", "emoji": "🗄️",
          "why": "A database reached from the network — serious if it's exposed.",
          "how": "If this is a developer's test machine, the database should only answer on 'localhost'."},
    5432: {"name": "PostgreSQL database", "emoji": "🗄️",
          "why": "A database reached from the network — serious if it's exposed.",
          "how": "The database should only answer on 'localhost', not the whole Wi-Fi."},
    27017: {"name": "MongoDB database", "emoji": "🗄️",
          "why": "A database that historically ships without a password.",
          "how": "Turn on authentication and bind it to 'localhost', or this is a risk."},
    6379: {"name": "Redis data store", "emoji": "🗄️",
          "why": "A fast data store that is frequently probed unauthenticated.",
          "how": "Require a password and bind to 'localhost' unless truly needed."},
    1433: {"name": "SQL Server database", "emoji": "🗄️",
          "why": "A database reached from the network.",
          "how": "Keep SQL Server bound to 'localhost' unless apps need it across machines."},
    161: {"name": "Network monitoring (SNMP)", "emoji": "📡",
          "why": "Lets a stranger read device stats — some versions leak settings.",
          "how": "Turn off SNMP on the device or change its community string from 'public'."},
    8000: {"name": "Web / dev server", "emoji": "🌐",
          "why": "A spare web port often left open by developer tools.",
          "how": "Stop the dev server when not coding on it."},
    8080: {"name": "Web / dev server", "emoji": "🌐",
          "why": "A spare web port often left open by developer tools.",
          "how": "Stop the dev server when not coding on it."},
    8888: {"name": "Web / dev server", "emoji": "🌐",
          "why": "A spare web port often left open by developer tools.",
          "how": "Stop the dev server when not coding on it."},
}

_FALLBACK_DOOR = {
    "name": "A program listening on this port",
    "emoji": "⚙️",
    "why": "Something on that device accepts connections from the Wi-Fi.",
    "how": "On the device, check which program is using this port and disable it if you "
           "don't use it (netstat / activity monitor, or the app's own settings).",
}


def fix_howtos(df) -> dict:
    """Per-device "open door" list with a close-guide for each service."""
    devices = []
    for _, row in df.iterrows():
        raw = str(row.get("services", ""))
        ports = [int(p) for p in raw.split(",") if str(p).strip().isdigit()]
        if not ports:
            continue
        doors = []
        for port in ports:
            info = SERVICE_DOORS.get(port, _FALLBACK_DOOR)
            doors.append({"port": port, "name": info["name"], "emoji": info["emoji"],
                          "why": info["why"], "how": info["how"]})
        devices.append({
            "ip": str(row.get("ip", "?")),
            "vendor": str(row.get("vendor", "") or "") or str(row.get("os_guess", "")) or "unknown device",
            "doors": doors,
        })
    return {"devices": devices}


def quiet_network_forecast(model, n_windows: int = 20, seed: int = 7,
                           context_length: int = 16, k_steps: int = 10) -> dict:
    """The AI's read of routine traffic: no probing, no open doors.

    Uses the same replay pipeline as the simulated attack but with all-benign
    windows, so the result shows how quiet the network looks once there is
    nothing for a stranger to grab.
    """
    phases = [0] * int(n_windows)
    out = simulate_attack(model, phases, seed=seed,
                          context_length=context_length, k_steps=k_steps)
    if "error" in out:
        return out
    probs = np.array([t["prob"] for t in out["timeline"]])
    return {
        "peak": float(probs.max()),
        "mean": float(probs.mean()),
        "probabilities": probs,
    }


# ─── Counterfactual "what-if" simulation ─────────────────────────────────────
# Runs the same attack twice: once untouched (baseline) and once with a defence
# action applied to the generated flows. The AI then shows how much the peak
# intrusion confidence drops, i.e. it *recommends* an action and proves the
# effect in plain words.

_LOGIN_PORTS = (22, 21, 3389)  # ssh / ftp / rdp — the brute-force targets
# The external hosts the traffic generators use as attackers (RPF-1918 doc net).
_ATTACKER_IPS = ("198.51.100.7", "203.0.113.42", "192.0.2.99")


def _block_port_transform(port: int):
    port = int(port)

    def transform(flows):
        if flows is None or flows.empty:
            return flows
        if "Dst Port" not in flows.columns:
            return flows
        return flows[flows["Dst Port"] != port]

    return transform


def _disconnect_host_transform(ip: str):
    ip = str(ip)

    def transform(flows):
        if flows is None or flows.empty:
            return flows
        mask = (flows["Src IP"] == ip) | (flows["Dst IP"] == ip)
        return flows[~mask]

    return transform


def _isolate_attacker_transform(ips):
    if isinstance(ips, str):
        ips = [ips]
    ips = set(str(x) for x in ips)

    def transform(flows):
        if flows is None or flows.empty:
            return flows
        mask = (flows["Src IP"].isin(ips)) | (flows["Dst IP"].isin(ips))
        return flows.loc[~mask]

    return transform


def _slow_logins_transform(factor: float):
    factor = float(np.clip(factor, 0.05, 1.0))
    rng = np.random.RandomState(0)  # deterministic throttling for reproducibility

    def transform(flows):
        if flows is None or flows.empty:
            return flows
        mask = flows["Dst Port"].isin(_LOGIN_PORTS).to_numpy()
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return flows
        n_keep = max(1, int(round(idx.size * factor)))
        keep_pos = set(rng.choice(idx, size=n_keep, replace=False).tolist())
        drop = mask.copy()
        drop[list(keep_pos)] = False
        return flows.loc[~drop]

    return transform


def _attack_targets(phases: List[int], seed: int, window_size_s: float):
    """Replay the generators (no model) and tally which ports/IPs got hit.

    Runs with the exact same seeded RNG as ``simulate_attack`` so the "auto"
    intervention targets match what the counterfactual replay will generate.
    """
    import scripts.train_live as train_live
    rng = np.random.RandomState(seed)
    train_live._RNG = rng
    ports: Counter = Counter()
    ips: Counter = Counter()
    sources: Counter = Counter()
    t0 = 1_700_000_000.0
    for w, phase in enumerate(phases):
        if not phase:
            continue
        agg = _make_aggregator()
        _GENERATORS[phase](agg, t0 + w * window_size_s, rng)
        flows = agg.snapshot()
        if flows is None or flows.empty:
            continue
        if "Dst Port" in flows.columns:
            ports.update(int(p) for p in flows["Dst Port"] if 0 < int(p) < 65536)
        if "Dst IP" in flows.columns:
            ips.update(str(i) for i in flows["Dst IP"])
        if "Src IP" in flows.columns:
            sources.update(str(i) for i in flows["Src IP"])
    return ports, ips, sources


def _normalise_intervention(intervention: Dict, phases: List[int],
                            seed: int, window_size_s: float) -> Dict:
    """Resolve a what-if spec into (transform, plain, label) or None."""
    spec = intervention or {}
    itype = spec.get("type")

    if itype == "block_port":
        port = spec.get("port") or "auto"
        if port in ("auto", None):
            counts, _, _ = _attack_targets(phases, seed, window_size_s)
            if not counts:
                return None
            port = counts.most_common(1)[0][0]
        port = int(port)
        return {
            "transform": _block_port_transform(port),
            "plain": f"closing the network door on port {port}",
            "label": f"Block port {port}",
        }

    if itype == "disconnect_host":
        ip = spec.get("ip") or "auto"
        if ip in ("auto", None):
            _, counts, _ = _attack_targets(phases, seed, window_size_s)
            if not counts:
                return None
            ip = counts.most_common(1)[0][0]
        ip = str(ip)
        return {
            "transform": _disconnect_host_transform(ip),
            "plain": f"disconnecting {ip} from the network",
            "label": f"Disconnect {ip}",
        }

    if itype == "isolate_attacker":
        src = spec.get("source") or "auto"
        if src in ("auto", None):
            _, _, sources = _attack_targets(phases, seed, window_size_s)
            external = sorted(ip for ip in sources if not ip.startswith("10."))
            ips = list(_ATTACKER_IPS) if not external else sorted(set(external).intersection(_ATTACKER_IPS)) or list(_ATTACKER_IPS)
        else:
            ips = [str(src)]
        tag = ", ".join(ips[:2]) + ("..." if len(ips) > 2 else "")
        return {
            "transform": _isolate_attacker_transform(ips),
            "plain": f"isolating the external source {tag} at your gateway",
            "label": f"Isolate attacker ({tag})",
        }

    if itype == "slow_logins":
        factor = float(spec.get("factor", 0.3))
        return {
            "transform": _slow_logins_transform(factor),
            "plain": f"slowing the attacker's login attempts by {100 * (1 - factor):.0f}%",
            "label": "Slow login attempts",
        }

    if itype == "do_nothing":
        return {"transform": lambda f: f, "plain": "doing nothing", "label": "Do nothing"}

    return None


def simulate_intervention(
    model,
    intervention: Dict,
    phases: List[int] = None,
    seed: int = 7,
    context_length: int = 16,
    window_size_s: float = 5.0,
    k_steps: int = 10,
) -> dict:
    """Replay the attack twice and show what the defence action buys you.

    Returns a dict with ``baseline`` and ``counterfactual`` peaks/probabilities,
    the ``reduction`` in peak confidence, the resolved ``intervention`` (incl.
    auto-detected targets), and a plain-language ``verdict.story``.
    """
    if _GENERATORS is None:
        return {"error": "Could not load traffic generators (scripts/train_live.py)."}
    if phases is None:
        phases = build_phase_plan()

    baseline = simulate_attack(model, phases, seed=seed,
                               context_length=context_length,
                               window_size_s=window_size_s, k_steps=k_steps)
    if "error" in baseline:
        return baseline

    spec = _normalise_intervention(intervention, phases, seed, window_size_s)
    if spec is None:
        return {"baseline": dict(baseline),
                "error": f"Unsupported intervention: {intervention}"}

    counter = simulate_attack(model, phases, seed=seed,
                              context_length=context_length,
                              window_size_s=window_size_s, k_steps=k_steps,
                              flow_transform=spec["transform"])
    if "error" in counter:
        return {"baseline": dict(baseline), "error": counter["error"]}

    base_probs = np.array([t["prob"] for t in baseline["timeline"]], dtype=float)
    ctr_probs = np.array([t["prob"] for t in counter["timeline"]], dtype=float)
    peak_base = float(base_probs.max()) if base_probs.size else 0.0
    peak_ctr = float(ctr_probs.max()) if ctr_probs.size else 0.0
    mean_base = float(base_probs.mean()) if base_probs.size else 0.0
    mean_ctr = float(ctr_probs.mean()) if ctr_probs.size else 0.0
    reduction = mean_base - mean_ctr

    story = [
        f"Without any action the simulation's average intrusion confidence sits "
        f"at {mean_base:.0%}.",
        f"If you respond by {spec['plain']}, that average drops from {mean_base:.0%} "
        f"to {mean_ctr:.0%} — {(reduction / mean_base):.0%} lower.",
    ]
    if reduction >= 0.08:
        story.append("Strong effect: this measurably shrinks the alarm window and gives "
                     "your response team room to breathe.")
    else:
        story.append("The action barely moves the needle — the model still sees hostile "
                     "traffic elsewhere. Isolate the external source at the gateway for a real effect.")

    return {
        "intervention": {
            "type": (intervention or {}).get("type"),
            "label": spec["label"],
            "plain": spec["plain"],
        },
        "reduction": reduction,
        "recommended": bool(reduction >= 0.08),
        "phases": [int(t["phase"]) for t in baseline["timeline"]],
        "baseline": {
            "peak": peak_base,
            "mean": mean_base,
            "stage_name": baseline["stage_name_peak"],
            "probabilities": base_probs.tolist(),
            "story": baseline["story"],
        },
        "counterfactual": {
            "peak": peak_ctr,
            "mean": mean_ctr,
            "stage_name": counter["stage_name_peak"],
            "probabilities": ctr_probs.tolist(),
            "story": counter["story"],
        },
        "verdict": {"story": story},
    }


def forecast_from_scan(
    df,
    probed_ports: List[int],
    model,
    ctx_win: int = 12,
    attack_win: int = 1,
    k_steps: int = 10,
    seed: int = None,
) -> dict:
    """Feed a completed scan through the live world-model pipeline.

    The scanner's own probe flows are replayed so the model sees the
    reconnaissance signature it just generated, then a K-step rollout predicts
    how far the attack would get. Returns a dict with ``probabilities``,
    ``stages`` (names), ``peak``, ``stage_display``, or an ``error`` key.
    """
    from scapy.all import get_if_addr
    from src.data.network_scanner import COMMON_PORTS, emit_probe_flows
    from src.data.pcap_parser import FlowAggregator
    from src.data.live_capture import LiveStateMapper, OnlineScaler

    try:
        src_ip = get_if_addr() or "10.10.10.10"
    except Exception:  # noqa: BLE001
        src_ip = "10.10.10.10"

    win_s = 5.0
    base = 1_700_000_000.0
    agg = FlowAggregator()
    emit_probe_flows(df, agg, src_ip=src_ip, benign_before=int(ctx_win),
                     seed=int(seed if seed is not None else _default_seed()),
                     window_size_s=win_s, t0=base,
                     attack_windows=int(attack_win),
                     probed_ports=[int(p) for p in (probed_ports or list(COMMON_PORTS))])

    flows = agg.snapshot()
    mapper = LiveStateMapper()
    scaler = OnlineScaler(STATE_DIM)
    states, _next = extract_window_states(flows, base, win_s, mapper, scaler, 0)
    if len(states) < 2:
        return {"error": "Not enough flow detail to form a state sequence. Re-run the scan or use a lighter port list."}
    ctx = np.asarray(states[-int(ctx_win):], dtype=np.float32)
    rollout = _rollout(model, ctx, int(k_steps))
    if rollout is None:
        return {"error": "Forecast failed — model returned no rollout."}
    probs = np.asarray(rollout["probabilities"])
    stage_ids = np.asarray(rollout["stages"])
    peak = float(probs.max())
    stage_display = STAGE_NAMES[int(stage_ids[int(probs.argmax())])]
    return {
        "probabilities": probs,
        "stages": stage_ids,
        "seq": [STAGE_NAMES[int(s)] for s in stage_ids],
        "peak": peak,
        "stage_display": stage_display,
    }


def _default_seed() -> int:
    import random
    return int(random.randint(0, 2 ** 31))


# ── tiny internal helpers (kept import-light) ────────────────────────────────

def _make_aggregator():
    from src.data.pcap_parser import FlowAggregator
    return FlowAggregator()


def _rollout(model, ctx: np.ndarray, k_steps: int):
    import torch
    from src.prediction.rollout_engine import RolloutEngine
    engine = RolloutEngine(model)
    tensor = torch.tensor(ctx[None, :, :], dtype=torch.float32)
    return engine.rollout(tensor, k_steps=k_steps)