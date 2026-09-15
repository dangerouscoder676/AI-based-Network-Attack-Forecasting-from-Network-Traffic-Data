"""NetSeer — command-line network scanner.

Active host discovery + light TCP connect scan + attack-surface scoring /
MITRE mapping, runnable headlessly for the judges' demo or cron monitoring.

Usage::

    python scripts/scan_network.py --network 192.168.29.0/24
    python scripts/scan_network.py --network 10.0.0.0/24 --method icmp \\
        --ports 22,80,443 --json results/scan.json
    python scripts/scan_network.py --forecast         # + world model forecast

Requires Npcap for ARP discovery; ICMP (`ping`) and TCP connect work without it.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.network_scanner import (
    discover_hosts,
    guess_local_network,
    host_risk,
    network_cidr_via_scapy,
    network_risk_summary,
    scan_ports,
    emit_probe_flows,
    COMMON_PORTS,
)


def _progress(fraction):
    bar_len = 40
    filled = int(bar_len * fraction)
    print(f"\r  [{'#' * filled}{' ' * (bar_len - filled)}] {fraction:5.1%}",
          end="", flush=True)


def run_forecast(df: pd.DataFrame, src_ip: str, k_steps: int = 10,
                 probed_ports: list = None, attack_windows: int = 1) -> dict:
    """K-step rollout of the live World Model over the scanner's own probes."""
    from src.models.world_model import WorldModel
    from src.prediction.rollout_engine import RolloutEngine
    from src.utils.config import load_config
    from src.data.live_capture import LiveStateMapper, OnlineScaler, extract_window_states, STATE_DIM
    from src.data.pcap_parser import FlowAggregator
    from src.data.network_scanner import emit_probe_flows

    cfg = load_config("config.yaml").as_dict()
    try:
        model = WorldModel.from_config(cfg)
        model.load_state_dict(torch_load("best_live_model.pth"))
    except Exception:  # noqa: BLE001
        import torch
        model = WorldModel.from_config(cfg)
        path = next(p for p in ["models/best_live_model.pth", "models/live_transformer_model.pth",
                                "models/best_model.pth"] if os.path.exists(p))
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()

    agg = FlowAggregator()
    ctx_win = 12
    win_s = 5.0
    base = 1_700_000_000.0

    emit_probe_flows(df, agg, src_ip=src_ip, benign_before=ctx_win,
                     seed=7, window_size_s=win_s, t0=base,
                     attack_windows=attack_windows, probed_ports=probed_ports)

    flows = agg.snapshot()
    mapper = LiveStateMapper()
    scaler = OnlineScaler(STATE_DIM)
    states, _ = extract_window_states(flows, base, win_s, mapper, scaler, 0)
    if len(states) < 2:
        return {"error": "insufficient states"}
    ctx = np.stack(states[-int(cfg["data"]["context_length"]):]).astype(np.float32)
    rollout = RolloutEngine(model).rollout(torch.tensor(ctx[None], dtype=torch.float32), k_steps=k_steps)
    return {
        "probabilities": [round(float(p), 4) for p in rollout["probabilities"]],
        "stages": [int(s) for s in rollout["stages"]],
        "stage_names": rollout["stage_names"],
        "peak": float(np.max(rollout["probabilities"])),
    }


def torch_load(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--network", default=None, help="CIDR to scan (default: auto-detect)")
    parser.add_argument("--method", choices=["auto", "arp", "icmp"], default="auto")
    parser.add_argument("--ports", default=None, help="Comma-separated ports (default: curated common set)")
    parser.add_argument("--timeout", type=float, default=0.8, help="Per-port connect timeout (s)")
    parser.add_argument("--json", default=None, help="Write results as JSON to this path")
    parser.add_argument("--forecast", action="store_true", help="Also run the world-model forecast")
    args = parser.parse_args()

    network = args.network or network_cidr_via_scapy() or guess_local_network()
    if network in ("0.0.0.0/0", None):
        print("Could not auto-detect the local network; pass --network CIDR")
        return 1
    ports = [int(p) for p in (args.ports or "").split(",") if p] or list(COMMON_PORTS)

    print(f"Network Scout — scanning {network} (method={args.method}, {len(ports)} ports)")
    t0 = time.time()
    hosts = discover_hosts(network, prefer=args.method, progress=_progress)
    print(f"\n  Discovery: {len(hosts)} live hosts in {time.time()-t0:.1f}s")
    if not hosts:
        print("  No hosts found.")
    scan_ports(hosts, ports=ports, timeout_s=args.timeout, progress=_progress)
    print()
    df = host_risk(hosts)
    total = len([1 for _ in __import__("ipaddress").ip_network(network, strict=False).hosts()])
    summary = network_risk_summary(df, total)

    print(f"\n  Attack surface: {summary['level']} "
          f"(score {summary['risk_score']:.1f}, {summary['exposed_hosts']}/{summary['hosts']} exposed, "
          f"{summary['open_ports']} open ports)")
    for m in summary["mitre"]:
        print(f"    MITRE: {m}")
    pd.set_option("display.max_columns", None)
    print(df[["ip", "mac", "vendor", "os_guess", "open_ports", "services",
              "risk_label", "mitre_techniques"]].to_string(index=False))

    result = {"network": network, "summary": summary,
              "hosts": df.where(pd.notnull(df), None).to_dict(orient="records"),
              "generated": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if args.forecast and df is not None and not df.empty:
        import socket
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            src_ip = s.getsockname()[0]
            s.close()
        except OSError:
            src_ip = "127.0.0.1"
        print("\n  Running world-model forecast over scan probes…")
        forecast = run_forecast(df, src_ip=src_ip, probed_ports=ports)
        if "error" not in forecast:
            seq = " → ".join(dict.fromkeys(forecast["stage_names"]))
            print(f"  Peak infiltration probability: {forecast['peak']:.1%}")
            print(f"  ATT&CK progression: {seq}")
        result["forecast"] = forecast

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())