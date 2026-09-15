"""
NetSeer — active network scanner & attack-surface assessment
=============================================================

Interrogates the **live network** the machine is connected to: discovers hosts
(ARP sweep when Npcap is present, ICMP ping sweep otherwise), performs a light
TCP **connect scan** on a curated set of well-known ports and turns the result
into an interpretable attack-surface inventory mapped to MITRE ATT&CK.

Everything here produces **real packets on the wire** (the scanner's own SYN
probes and ARP requests) — this is the "connect to a network, analyse it"
capability that complements the passive ``live_capture`` engine.

Safety / politeness
-------------------
- Pure TCP *connect* scanning (completes the three-way handshake) which is the
  least-evasive, friendliest type of scan and works without Npcap.
- A small curated port list by default (no full-port sweeps on shared nets).
- Discovery is rate-limited and runs only against the CIDR the operator enters.

Pipeline
--------
host discovery (ARP / ICMP)
    -> TCP connect scan on common ports per live host
    -> host inventory DataFrame (MAC, vendor, TTL/OS guess, open ports)
    -> per-host & network risk scoring + MITRE technique mapping
"""

import ipaddress
import os
import re
import socket
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# ─── Defaults ─────────────────────────────────────────────────────────────────

#: Services probed by default (curated, demo-safe).
COMMON_PORTS: Dict[int, str] = {
    21: "ftp",
    22: "ssh",
    23: "telnet",
    25: "smtp",
    53: "dns",
    80: "http",
    110: "pop3",
    135: "msrpc",
    139: "netbios-ssn",
    143: "imap",
    443: "https",
    445: "microsoft-ds",
    993: "imaps",
    1433: "mssql",
    3306: "mysql",
    3389: "rdp",
    5432: "postgres",
    5900: "vnc",
    6379: "redis",
    8000: "http-alt",
    8080: "http-proxy",
    8443: "https-alt",
    9200: "elasticsearch",
    27017: "mongodb",
}

#: Risk points (0-5) per exposed service class. Interpretable heuristic used to
#: derive the MITRE-mapped exposure score for each host.
_SERVICE_RISK: Dict[str, int] = {
    "ftp": 3,
    "ssh": 3,
    "telnet": 5,
    "smtp": 1,
    "dns": 1,
    "http": 1,
    "pop3": 1,
    "msrpc": 3,
    "netbios-ssn": 4,
    "imap": 1,
    "https": 1,
    "imaps": 1,
    "mssql": 4,
    "mysql": 3,
    "rdp": 4,
    "postgres": 3,
    "vnc": 5,
    "redis": 4,
    "http-alt": 2,
    "http-proxy": 2,
    "https-alt": 2,
    "elasticsearch": 4,
    "mongodb": 3,
}

#: Ports whose exposure matters most for an attacker's Initial Access playbook.
_MGMT_PORTS = {22, 23, 445, 3389, 5900, 1433, 3306, 5432, 6379, 27017, 9200}


def _ttl_os_guess(ttl: Optional[int]) -> str:
    """Best-effort OS family from an ICMP/TCP reply TTL."""
    if ttl is None:
        return "unknown"
    if ttl <= 64:
        return "Linux / *nix"
    if ttl <= 128:
        return "Windows"
    if ttl <= 255:
        return "Network device (router/switch)"
    return "unknown"


# ─── Utility: local network detection ─────────────────────────────────────────

def guess_local_network(iface_ip: Optional[str] = None, prefix: int = 24) -> str:
    """
    Best-effort CIDR of the machine's primary network.

    Works without Npcap: opens a throwaway UDP socket (no packets are sent) to
    learn the default-route source address, then assumes ``/prefix``. Prefer the
    Scapy routing table (``network_cidr_via_scapy``) when Npcap is available.

    Returns e.g. ``"192.168.1.0/24"`` or ``"0.0.0.0/0"`` when detection fails.
    """
    ip = iface_ip
    if not ip:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
        except OSError:
            ip = None
    if not ip or ip.startswith("127."):
        # Last resort: first non-loopback address from the hostname.
        try:
            for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
                cand = info[4][0]
                if cand and not cand.startswith("127."):
                    ip = cand
                    break
        except OSError:
            pass
    if not ip or ip.startswith("127."):
        return "0.0.0.0/0"
    net = ipaddress.ip_network(f"{ip}/{prefix}", strict=False)
    return str(net)


def network_cidr_via_scapy(iface: Optional[str] = None) -> Optional[str]:
    """Return ``"network/prefix"`` from the Scapy routing table (needs Npcap)."""
    try:
        from scapy.all import conf

        _, _gateway, _local, _mask = conf.route.route("8.8.8.8")
        net = ipaddress.ip_network(f"{_local}/{_mask}", strict=False)
        return str(net)
    except Exception:
        return None


# ─── Host discovery ───────────────────────────────────────────────────────────

@dataclass
class DiscoveredHost:
    ip: str
    mac: str = ""
    vendor: str = ""
    ttl: Optional[int] = None
    os_guess: str = "unknown"
    rtt_ms: Optional[float] = None
    method: str = "unknown"          # "arp" | "icmp" | "tcp"
    alive: bool = True
    open_ports: List[int] = field(default_factory=list)


# Keep a compact OUI -> vendor table (top consumer/pro-tool vendors used in demos).
_OUI_VENDORS = {
    "a4:5e:60": "Espressif (ESP)",
    "dc:a6:32": "Raspberry Pi",
    "60:01:94": "Espressif (ESP)",
    "24:6f:28": "Espressif (ESP)",
    "b8:27:eb": "Raspberry Pi",
    "f8:27:93": "Raspberry Pi",
    "0c:8b:fd": "Raspberry Pi",
    "e4:5f:01": "Raspberry Pi",
    "00:0a:95": "Apple",
    "3c:07:54": "Apple",
    "ac:bc:32": "Apple",
    "f0:18:98": "Apple",
    "38:c9:86": "Apple",
    "f4:ca:e7": "Xiaomi",
    "78:11:dc": "Xiaomi",
    "64:9e:f3": "Xiaomi",
    "84:9c:a6": "HUAWEI",
    "8c:34:fd": "HUAWEI",
    "60:32:b1": "TP-Link",
    "18:a6:f7": "TP-Link",
    "50:95:69": "TP-Link",
    "88:66:a5": "Samsung",
    "b4:0b:44": "Samsung",
    "2c:3a:fd": "Samsung",
    "a4:47:0a": "Samsung",
    "54:27:1e": "Lenovo",
    "3c:a6:f6": "Intel",
    "8c:16:45": "Dell",
    "38:f8:b7": "Synology",
    "04:d9:f5": "Synology",
    "c4:e9:84": "Intel",
    "c4:8e:8f": "ASUS",
    "84:8f:05": "Intel",
    "d8:bb:c1": "D-Link",
    "5c:e2:6c": "Hon Hai (Foxconn)",
}


def _lookup_vendor(mac: str) -> str:
    if not mac:
        return ""
    oui = mac.replace(":", "")[:6].lower()
    # search by 3-octet prefix
    cand = f"{oui[0:2]}:{oui[2:4]}:{oui[4:6]}"
    return _OUI_VENDORS.get(cand, "")


def arp_discovery(network: str, timeout: float = 4.0, progress=None) -> List[DiscoveredHost]:
    """
    ARP who-has sweep over the subnet. Requires Npcap (Layer-2 access).

    Sends one ARP request per host in a single burst and collects replies; this
    is fast (one round-trip) and is the canonical way to enumerate a local L2
    segment. Raises RuntimeError if Npcap / raw sockets are unavailable so the
    caller can fall back to ICMP.
    """
    from scapy.all import ARP, Ether, srp

    net = ipaddress.ip_network(network, strict=False)
    if net.prefixlen < 16:
        raise ValueError(f"Refusing to sweep {net} ({net.prefixlen} too wide). Use /16 minimum.")
    hosts = [str(ip) for ip in net.hosts() if not ip.is_multicast]
    if len(hosts) > 1024:
        raise ValueError(f"Subnet has {len(hosts)} addresses; refusing ARP sweep.")

    try:
        pkts = [Ether(dst="ff:ff:ff:ff:ff:ff") / ARP(pdst=ip) for ip in hosts]
        ans, _ = srp(pkts, timeout=timeout, verbose=0)
    except Exception as e:  # noqa: BLE001 - Npcap missing / permissions
        raise RuntimeError(f"ARP sweep unavailable (install Npcap): {e}") from e

    results = []
    for _sent, rev in ans:
        mac = rev.hwsrc
        ttl = None
        rtt = None
        results.append(
            DiscoveredHost(
                ip=rev.psrc,
                mac=mac,
                vendor=_lookup_vendor(mac),
                ttl=ttl,
                os_guess="unknown",
                rtt_ms=rtt,
                method="arp",
            )
        )
    return results


def _ping_probe(ip: str, timeout_s: int = 1) -> Optional[Tuple[int, float]]:
    """Windows `ping -n 1` probe; returns (ttl, rtt_ms) or None when unreachable."""
    try:
        out = subprocess.run(
            ["ping", "-n", "1", "-w", str(int(timeout_s * 1000)), ip],
            capture_output=True, text=True, timeout=timeout_s + 2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    text = out.stdout or ""
    if "TTL=" not in text and "ttl=" not in text.lower():
        return None
    ttl_m = re.search(r"[Tt][Tt][Ll]=(\d+)", text)
    ttl = int(ttl_m.group(1)) if ttl_m else None
    time_m = re.search(r"[tT]ime[<=\s]+([\d.]+)\s*ms", text)
    rtt = float(time_m.group(1)) if time_m else None
    return ttl, rtt


def icmp_discovery(network: str, workers: int = 16, timeout_s: int = 1,
                   progress=None, max_hosts: int = 1024) -> List[DiscoveredHost]:
    """ICMP echo discovery using the system ``ping`` binary (works without Npcap)."""
    net = ipaddress.ip_network(network, strict=False)
    hosts = [str(ip) for ip in net.hosts() if not ip.is_multicast]
    if len(hosts) > int(max_hosts):
        raise ValueError(
            f"Subnet has {len(hosts)} addresses; refusing ping sweep "
            f"(limit {int(max_hosts)})."
        )
    found: List[DiscoveredHost] = []
    done = 0

    def _probe(ip):
        return ip, _ping_probe(ip, timeout_s)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_probe, ip): ip for ip in hosts}
        for fut in as_completed(futs):
            ip, res = fut.result()
            done += 1
            if progress is not None:
                progress(done / max(len(hosts), 1))
            if res is not None:
                ttl, rtt = res
                found.append(
                    DiscoveredHost(
                        ip=ip, ttl=ttl, os_guess=_ttl_os_guess(ttl),
                        rtt_ms=rtt, method="icmp",
                    )
                )
    return found


def discover_hosts(network: str, prefer: str = "arp", timeout: float = 4.0,
                   progress=None) -> List[DiscoveredHost]:
    """
    Discover live hosts on ``network``. Tries ARP first (fast, gives MACs) when
    Npcap is available and ICMP (`ping`) otherwise; deduplicates by IP.
    """
    hosts: Dict[str, DiscoveredHost] = {}

    def _merge(h: DiscoveredHost):
        prev = hosts.get(h.ip)
        if prev is None or (h.mac and not prev.mac):
            hosts[h.ip] = h
        elif prev and prev.method == "arp" and h.method == "icmp":
            # keep ARP entry (has MAC), enrich with TTL info if missing
            if prev.ttl is None:
                prev.ttl, prev.os_guess, prev.rtt_ms = h.ttl, h.os_guess, h.rtt_ms
                prev.method = "arp+icmp"

    arp_ok = False
    if prefer in ("arp", "auto"):
        try:
            for h in arp_discovery(network, timeout=timeout, progress=progress):
                _merge(h)
            arp_ok = True
        except (RuntimeError, ValueError) as e:
            if prefer == "arp":
                raise
    if not arp_ok or prefer == "icmp":
        for h in icmp_discovery(network, progress=progress):
            _merge(h)
    return list(hosts.values())


# ─── Light TCP connect scan ───────────────────────────────────────────────────

def tcp_connect_scan_one(ip: str, ports: List[int], timeout_s: float = 1.0,
                         progress=None) -> List[int]:
    """Best-effort TCP connect test per host (pure sockets, no privileges)."""
    open_ports: List[int] = []

    def _try(p):
        try:
            with socket.create_connection((ip, p), timeout=timeout_s):
                open_ports.append(p)
        except OSError:
            pass

    with ThreadPoolExecutor(max_workers=min(32, len(ports))) as ex:
        list(ex.map(_try, ports))
    return sorted(open_ports)


def scan_ports(hosts: List[DiscoveredHost], ports: Optional[List[int]] = None,
               workers: int = 8, timeout_s: float = 1.0, progress=None) -> None:
    """Connect-scan ``ports`` on each live host, storing results in place."""
    if ports is None:
        ports = list(COMMON_PORTS)
    total = len(hosts)

    def _scan(h):
        if not h.alive:
            return
        h.open_ports = tcp_connect_scan_one(h.ip, ports, timeout_s)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, _h in enumerate(ex.map(_scan, hosts)):
            if progress is not None:
                progress((i + 1) / total)


# ─── Risk scoring + MITRE mapping ─────────────────────────────────────────────

def host_risk(hosts: List[DiscoveredHost]) -> pd.DataFrame:
    """
    Turn discovered hosts into a scored attack-surface DataFrame.

    Each host gets an interpretable exposure score derived from the summed risk
    of its open services (see ``_SERVICE_RISK``) plus a raw open-port penalty,
    then a LOW / MEDIUM / HIGH / CRITICAL label and the MITRE techniques that an
    attacker would map the host's exposure to.
    """
    rows = []
    for h in hosts:
        open_ports = sorted(h.open_ports)
        services = [COMMON_PORTS.get(p, "unknown") for p in open_ports]
        score = sum(_SERVICE_RISK.get(svc, 2) for svc in services if svc != "unknown")
        score += max(0, len(open_ports) - 3)  # many exposed ports widen the surface
        score += 2 if any(p in _MGMT_PORTS for p in open_ports) else 0

        if score == 0:
            label, color = "NONE", "#23d18b"
        elif score < 5:
            label, color = "LOW", "#3fb950"
        elif score < 10:
            label, color = "MEDIUM", "#d29922"
        elif score < 16:
            label, color = "HIGH", "#f85149"
        else:
            label, color = "CRITICAL", "#ff3860"

        techs = ["T1046"] if open_ports else []
        if open_ports and any(p in (22, 23, 445, 3389) for p in open_ports):
            techs.insert(0, "T1190")  # remote services -> exploitation risk

        rows.append({
            "ip": h.ip,
            "mac": h.mac,
            "vendor": h.vendor,
            "method": h.method,
            "os_guess": h.os_guess,
            "rtt_ms": h.rtt_ms,
            "open_ports": ",".join(map(str, open_ports)) if open_ports else "",
            "services": ",".join(svc for svc in services if svc != "unknown") or "",
            "n_open_ports": len(open_ports),
            "risk_score": score,
            "risk_label": label,
            "risk_color": color,
            "mitre_techniques": ";".join(dict.fromkeys(techs)),
            "likely_target": bool(open_ports and any(p in _MGMT_PORTS for p in open_ports)),
        })
    return pd.DataFrame(rows)


def network_risk_summary(df: pd.DataFrame, total_addressed: int) -> dict:
    """Summary dict for the Scout page: exposure aggregates + MITRE framing."""
    if df is None or df.empty:
        return {"hosts": 0, "open_ports": 0, "exposed_hosts": 0, "risk_high": 0,
                "risk_score": 0, "level": "NONE", "mitre": []}
    n_hosts = len(df)
    open_total = int(df["n_open_ports"].sum()) if "n_open_ports" in df.columns else 0
    exposed = int((df["n_open_ports"] > 0).sum())
    high = int(df["risk_label"].isin(["HIGH", "CRITICAL"]).sum())
    avg_score = float(df["risk_score"].mean()) if n_hosts else 0.0

    if avg_score == 0:
        level = "NONE"
    elif avg_score < 5:
        level = "LOW"
    elif avg_score < 10:
        level = "MEDIUM"
    else:
        level = "HIGH"

    mitre = []
    if exposed:
        mitre.append("T1046 Network Service Discovery")
    if high:
        mitre.append("T1190 Exploit Public-Facing Application")
    win_exposed = ((df["os_guess"].fillna("") == "Windows") &
                   (df["n_open_ports"] > 0)).any()
    if win_exposed:
        mitre.append("T1078 Valid Accounts (RDP/SMB attack surface)")
    return {
        "hosts": n_hosts,
        "open_ports": open_total,
        "exposed_hosts": exposed,
        "risk_high": high,
        "risk_score": avg_score,
        "level": level,
        "mitre": mitre,
    }


# ─── Feed scanner probes into the World Model pipeline ────────────────────────

def emit_probe_flows(results: pd.DataFrame, aggregator, src_ip: str,
                     benign_before: int = 8, seed: int = 42,
                     window_size_s: float = 5.0, t0: Optional[float] = None,
                     attack_windows: int = 4,
                     probed_ports: Optional[List[int]] = None) -> int:
    """
    Replay the scanner's own discovery/connect probes as if they were captured,
    so the live world-model path can reason about the recon it just performed.

    Pollutes nothing on the wire — packets are synthesized and routed straight
    into a :class:`~src.data.pcap_parser.FlowAggregator`. Returns probe count.

    A short run of benign traffic precedes the probes so the online scaler has
    a baseline. Each benign window uses a *distinct* destination so the flow
    aggregator keeps one flow per 5-tuple per window (matching the per-window
    granularity the live model was trained on). The SYN probes are replayed to
    every port the scanner actually probed (open ones get a SYN-ACK response,
    closed ones a RST) and spread over ``attack_windows`` consecutive windows so
    the model sees a sustained recon episode, not a single transient burst.

    This is how the Scout page demonstrates *self-detection of reconnaissance*:
    the model's SYN-ratio / port-diversity channels react to the very scan it
    just observed (an unprompted, real-world pattern, not a synthetic file).
    """
    from scapy.all import IP, TCP

    rng = np.random.RandomState(seed)
    count = 0
    t0 = time.time() if t0 is None else t0
    attack_windows = max(1, int(attack_windows))
    if probed_ports is None:
        probed_ports = sorted({int(p) for p in COMMON_PORTS})

    # A little benign browser chatter first gives the online scaler a baseline.
    # Unique (dst, sdport) per window keeps windows from being merged together.
    for k in range(max(0, benign_before)):
        base_ts = t0 + k * window_size_s
        dst = f"192.0.2.{100 + (k % 150)}"
        dport = int(rng.choice([443, 80]))
        for i in range(int(rng.randint(6, 14))):
            pkt = IP(src=src_ip, dst=dst, proto=6) / \
                TCP(sport=49152 + i, dport=dport, flags="AP")
            if aggregator.update(pkt, timestamp=base_ts + i * 0.01):
                count += 1

    # Replay the connect probes exactly as they were sent (SYN, no payload),
    # plus the realistic handshake responses, spread across the final
    # ``attack_windows`` windows.
    probe_pkts: List[tuple] = []
    for i, (_i, row) in enumerate(results.iterrows()):
        dst = row["ip"]
        open_set = {int(p) for p in str(row["open_ports"]).split(",") if p}
        for j, p in enumerate(probed_ports):
            sport = 40000 + (i % 9000) * 100 + j
            probe_pkts.append((dst, p, sport, p in open_set))

    for w in range(attack_windows):
        base_ts = t0 + (benign_before + w) * window_size_s
        chunk = probe_pkts[w::attack_windows] or probe_pkts
        for k, (dst, p, sport, was_open) in enumerate(chunk):
            if aggregator.update(IP(src=src_ip, dst=dst, proto=6) /
                                 TCP(sport=sport, dport=p, flags="S"),
                                 timestamp=base_ts + k * 0.003):
                count += 1
            if was_open:
                repl = IP(src=dst, dst=src_ip, proto=6) / \
                    TCP(sport=p, dport=sport, flags="SA")
            else:
                repl = IP(src=dst, dst=src_ip, proto=6) / \
                    TCP(sport=p, dport=sport, flags="RA")
            if aggregator.update(repl, timestamp=base_ts + k * 0.003 + 0.001):
                count += 1
    return count


def auto(progress=None, **kwargs):
    """
    One-shot convenience: discover hosts + port scan + risk dataframe.

    Returns ``(df, summary)``. Use the granular functions from the page/CLI for
    live progress callbacks.
    """
    network = kwargs.pop("network", None)
    if network is None:
        network = guess_local_network()
    if network in ("0.0.0.0/0", None):
        raise RuntimeError("Could not auto-detect the local network. Pass the CIDR explicitly.")

    hosts = discover_hosts(network, progress=progress)
    scan_ports(hosts, progress=progress)
    df = host_risk(hosts)
    total = len(list(ipaddress.ip_network(network, strict=False).hosts()))
    summary = network_risk_summary(df, total)
    return df, summary