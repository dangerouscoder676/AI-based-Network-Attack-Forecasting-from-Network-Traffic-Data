"""
NetSeer — live network capture & online state engine
=====================================================

Turns *raw packets arriving in real time* into the 76-dim state vectors the
World Model was trained on, without re-training and without batch PCA / scalers.

Design
------
- ``FlowAggregator`` (from ``pcap_parser``) converts packets to bi-directional
  flows incrementally — identical to the offline parser.
- ``OnlineScaler`` keeps running per-channel mean/std (Welford) so normalization
  never needs the whole trace.
- ``LiveStateMapper`` maps each time-window's flow statistics into the model's
  76-dim semantic state space. It deliberately targets the *same signal
  channels* the synthetic training data used (e.g. channel 10 = SYN ratio,
  channel 3 = packets/s, channel 74 = destination-port diversity), so a real
  scan / brute-force / beacon produces the same kind of deviation the model
  learned to flag.
- ``LiveTrafficEngine`` streams packets from three feeds:
    * a **live interface** (Scapy sniff, requires Npcap + capture rights)
    * a **PCAP replay** projected onto a compressed demo clock (no privileges)
    * an **on-device attack simulator** that synthesizes packets locally
      (no privileges, self-contained demo)

Everything is deterministic and explainable — there is no "learned" offline
preprocessing in the live path.
"""

import os
import time
import threading
import numpy as np
import pandas as pd
from typing import Any, Optional, List, Tuple

from scapy.all import IP, TCP, UDP

from src.data.pcap_parser import FlowAggregator

STATE_DIM = 76
PROTO_TCP = 6
PROTO_UDP = 17


# ─── Online standardization (Welford's algorithm) ───────────────────────────
class OnlineScaler:
    """Running per-channel standardization with a bounded-z clip."""

    def __init__(self, n_features: int, clip: float = 6.0):
        self.n_features = int(n_features)
        self.clip = float(clip)
        self.count = 0
        self.mean = np.zeros(self.n_features, dtype=np.float64)
        self.m2 = np.zeros(self.n_features, dtype=np.float64)  # sum of sq. deviations
        self.sd = np.ones(self.n_features, dtype=np.float64)

    def update(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        n, d = x.shape
        if n == 0 or d != self.n_features:
            return
        new_count = self.count + n
        delta = x - self.mean[None, :]
        new_mean = self.mean + np.sum(delta, axis=0) / new_count
        delta2 = x - new_mean[None, :]
        self.m2 += np.sum(delta * delta2, axis=0)
        self.count = new_count
        self.mean = new_mean
        self.sd = np.sqrt(self.m2 / max(self.count - 1, 1)) + 1e-6

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        z = (x - self.mean[None, :]) / self.sd[None, :]
        return np.clip(z, -self.clip, self.clip)

    @property
    def fitted(self) -> bool:
        return self.count >= 2


# ─── Window → model-state mapping ───────────────────────────────────────────
class LiveStateMapper:
    """
    Maps one temporal window of flow statistics into a 76-dim state vector.

    Channels align with the semantics the world model was trained against
    (synthetic CIC-fidelity states): high value == suspicious behavior.
    All statistics are computed directly from the aggregated flows, so the
    mapping is deterministic and does not depend on flow-CSV feature aliasing.
    """

    def __init__(self):
        self._eps = 1e-8

    @staticmethod
    def _series(flows_df: pd.DataFrame, name: str, default: float = 0.0) -> np.ndarray:
        if name in flows_df.columns:
            return flows_df[name].replace([np.inf, -np.inf], np.nan).fillna(default).to_numpy(dtype=np.float64)
        return np.full(len(flows_df), default, dtype=np.float64)

    def build_raw_state(self, flows_df: pd.DataFrame, window_size_s: float = 1.0) -> np.ndarray:
        """
        Build a log-transformed, un-standardized 76-dim state for the window.

        Callers run it through an :class:`OnlineScaler` to get the final vector.
        """
        raw = np.zeros(STATE_DIM, dtype=np.float64)
        if flows_df is None or flows_df.empty:
            return raw

        n = len(flows_df)
        eps = self._eps

        dur_s = self._series(flows_df, "Flow Duration") / 1e6
        pkts = self._series(flows_df, "total_packets")
        tot_fwd = self._series(flows_df, "Tot Fwd Pkts")
        tot_bwd = self._series(flows_df, "Tot Bwd Pkts")
        fwd_bytes = self._series(flows_df, "TotLen Fwd Pkts")
        bwd_bytes = self._series(flows_df, "TotLen Bwd Pkts")
        syn = self._series(flows_df, "SYN_Flag_Cnt")
        rst = self._series(flows_df, "RST_Flag_Cnt")
        ack = self._series(flows_df, "ACK_Flag_Cnt")

        syn_ratio = np.mean(syn / (pkts + eps))
        rst_ratio = np.mean(rst / (pkts + eps))
        ack_ratio = np.mean(ack / (pkts + eps))
        pkt_len_mean = np.mean(self._series(flows_df, "Pkt Len Mean"))
        iat_mean = np.mean(self._series(flows_df, "Flow IAT Mean")) / 1e6
        iat_std = np.mean(self._series(flows_df, "Flow IAT Std")) / 1e6
        dst_ports = float(flows_df["Dst Port"].nunique()) if "Dst Port" in flows_df.columns else 1.0
        src_ips = float(flows_df["Src IP"].nunique()) if "Src IP" in flows_df.columns else 1.0
        sum_pkts = float(pkts.sum())

        mapping = [
            np.mean(flows_df["total_bytes"] / (pkts + eps)),            # 0 bytes per packet
            np.mean(tot_fwd / (tot_bwd + eps)),                          # 1 fwd/bwd packet ratio
            np.mean(dur_s),                                              # 2 flow duration (s)
            sum_pkts / max(float(window_size_s), 1e-6),                  # 3 window packet rate
            pkt_len_mean,                                                 # 4 (negated later)
            float(fwd_bytes.sum()),                                       # 5 fwd volume
            float(bwd_bytes.sum()),                                       # 6 bwd volume
            dst_ports,                                                    # 7 dst-port diversity
            src_ips,                                                      # 8 src-IP diversity
            float(n),                                                     # 9 flow count
            syn_ratio,                                                    # 10 SYN ratio ← recon/scan
            rst_ratio,                                                    # 11 RST ratio
            iat_std,                                                      # 12 (negated later ← beacon)
            ack_ratio,                                                    # 13 ACK ratio
            sum_pkts,                                                     # 14 packet volume
            np.mean(fwd_bytes / (bwd_bytes + eps)),                       # 15 fwd/bwd byte ratio
            iat_mean,                                                     # 16 (negated later)
        ]

        for ch, val in enumerate(mapping):  # fills channels 0..16
            raw[ch] = val

        # "smaller ⇒ suspicious" channels encoded as a rising negative-log value
        raw[4] = -float(np.log1p(max(pkt_len_mean, 0.0)))
        raw[12] = -float(np.log1p(max(iat_std, 0.0)))
        raw[16] = -float(np.log1p(max(iat_mean, 0.0)))

        # Log-scale the volume/ratio channels so the OnlineScaler sees clean order statistics
        for ch in (0, 1, 3, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 74, 75):
            raw[ch] = float(np.log1p(max(raw[ch], 0.0)))

        raw[74] = float(np.log1p(max(dst_ports, 0.0)))   # reinforce port-diversity channel
        raw[75] = float(np.log1p(max(sum_pkts, 0.0)))    # aggregate volume channel
        return raw

    def state(self, flows_df: pd.DataFrame, scaler: OnlineScaler, window_size_s: float = 1.0) -> np.ndarray:
        """Return the standardized 76-dim state for one window (and update scaler)."""
        raw = self.build_raw_state(flows_df, window_size_s=window_size_s)
        scaler.update(raw.reshape(1, -1))
        return scaler.transform(raw).reshape(-1).astype(np.float32)




# ─── Traffic feeds ───────────────────────────────────────────────────────────
def list_interfaces() -> List[str]:
    """Best-effort list of capture interfaces (empty when Scapy can't enumerate)."""
    try:
        from scapy.all import get_if_list, conf
        names = [str(i) for i in get_if_list() if str(i)]
        friendly = [str(conf.iface)] if str(conf.iface) else []
        return sorted(set(friendly + names))
    except Exception:
        return []


class LiveTrafficEngine:
    """
    A background capture/replay/simulation feed that aggregates packets into
    flows. Packet arrival timestamps are projected onto a virtual ``sim_t``
    epoch (real time for interface capture, compressed for PCAP replay).
    """

    def __init__(self):
        self.aggregator = FlowAggregator()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self.running = False
        self.mode = None              # interface | replay | simulator
        self.source = ""
        self.error = ""
        self.sim_t0: Optional[float] = None
        self._stop = threading.Event()
        self.stats = {"packets": 0, "flows": 0}

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def _spawn(self, fn):
        self._stop.clear()
        self.running = True
        self._thread = threading.Thread(target=fn, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        try:
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=1.0)
        except Exception:
            pass
        self.running = False

    def _ingest(self, pkt, sim_t: Optional[float] = None) -> None:
        if self._stop.is_set():
            return
        if sim_t is None:
            sim_t = time.time()
        with self._lock:
            if self.sim_t0 is None:
                self.sim_t0 = sim_t
            if self.aggregator.update(pkt, timestamp=float(sim_t)):
                self.stats["packets"] += 1

    # ── feed 1: live interface ────────────────────────────────────────────────
    def start_interface(self, iface: Optional[str] = None):
        self.mode = "interface"
        self.source = iface or "default interface"

        def _capture():
            try:
                from scapy.all import sniff, conf
                target = iface or conf.iface
                sniff(iface=str(target), prn=self._ingest, store=False, stop_filter=lambda p: self._stop.is_set())
            except Exception as e:
                self.error = str(e)
            finally:
                self.running = False

        self._spawn(_capture)

    # ── feed 2: PCAP replay on a compressed demo clock ────────────────────────
    def start_replay(self, pcap_path: str, demo_seconds: float = 90.0,
                     realtime: bool = False, max_packets: int = 400_000) -> bool:
        if not os.path.exists(pcap_path):
            self.error = f"PCAP not found: {pcap_path}"
            return False
        self.mode = "replay"
        self.source = os.path.basename(pcap_path)

        def _replay():
            try:
                from scapy.all import PcapReader
                samples = []
                with PcapReader(pcap_path) as rdr:
                    for pkt in rdr:
                        if len(samples) >= max_packets:
                            break
                        samples.append((float(pkt.time), pkt))
                if not samples:
                    self.error = "PCAP contained no packets."
                    self.running = False
                    return

                t_first = samples[0][0]
                t_last = samples[-1][0]
                span = max(t_last - t_first, 1.0)
                t0 = time.time()
                prev_sim = t0
                for i, (t, pkt) in enumerate(samples):
                    if self._stop.is_set():
                        break
                    frac = (t - t_first) / span
                    sim_t = t0 + frac * demo_seconds
                    self._ingest(pkt, sim_t=sim_t)
                    if realtime:
                        dt = sim_t - prev_sim
                        if dt > 0:
                            time.sleep(dt)
                        prev_sim = sim_t
            except Exception as e:
                self.error = str(e)
            finally:
                self.running = False

        self._spawn(_replay)
        return True

    # ── feed 3: on-device attack simulator (no privileges needed) ────────────
    def inject_scan(self, target: str = "192.168.1.100",
                    src: str = "10.0.0.9", ports: int = 160, warmup: float = 4.0) -> int:
        """
        Synthesize a TCP port-scan against ``target`` and feed it through the
        aggregator as if it were captured. Returns the number of packets added.
        """
        t0 = time.time()
        count = 0
        with self._lock:
            if self.sim_t0 is None:
                self.sim_t0 = t0
        base_warm = 8  # a little benign chatter first
        for _ in range(base_warm):
            self._ingest(self._benign_packet(), sim_t=t0 + np.random.uniform(0, warmup))
        for port in range(1, ports + 1):
            sim_t = t0 + warmup + port * 0.05
            sport = 40000 + (port % 4000)
            pkt = IP(src=src, dst=target, proto=PROTO_TCP) / TCP(sport=sport, dport=port, flags="S")
            self._ingest(pkt, sim_t=sim_t)
            count += 1
            # respond RST to ~half so some flows complete per direction
            if port % 2 == 0:
                rst = IP(src=target, dst=src, proto=PROTO_TCP) / TCP(sport=port, dport=sport, flags="RA")
                self._ingest(rst, sim_t=sim_t + 0.004)
                count += 1
        return count

    @staticmethod
    def _benign_packet():
        tgt = "192.168.1.1"
        t = np.random.randint(2, 3)
        src_ip = ".".join(str(np.random.randint(1, 255)) for _ in range(4))
        sport = int(np.random.randint(10000, 60000))
        payload = bytes(np.random.randint(0, 256, size=int(np.random.randint(32, 256))))
        if t == 2:
            dport = 443
            pkt = IP(src=src_ip, dst=tgt, proto=PROTO_TCP) / TCP(sport=sport, dport=dport, flags="AP") / payload
        else:
            dport = 53
            pkt = IP(src=src_ip, dst=tgt, proto=PROTO_UDP) / UDP(sport=sport, dport=dport) / payload
        return pkt

    def inject_bruteforce(self, target: str = "10.0.0.50",
                          src: str = "198.51.100.7", warmup: float = 4.0) -> int:
        """Synthesize an SSH brute-force burst (login attempts ⇒ repeated SYNs)."""
        t0 = time.time()
        count = 0
        with self._lock:
            if self.sim_t0 is None:
                self.sim_t0 = t0
        sport = 55555
        for i in range(120):
            sim_t = t0 + warmup + i * 0.035
            pkt = IP(src=src, dst=target, proto=PROTO_TCP) / TCP(sport=sport + i, dport=22, flags="S")
            self._ingest(pkt, sim_t=sim_t)
            count += 1
            ack = IP(src=target, dst=src, proto=PROTO_TCP) / TCP(sport=22, dport=sport + i, flags="SA")
            self._ingest(ack, sim_t=sim_t + 0.006)
            count += 1
        return count

    # ── queries ───────────────────────────────────────────────────────────────
    def snapshot_flows(self) -> pd.DataFrame:
        with self._lock:
            return self.aggregator.snapshot()

    def status(self) -> dict:
        with self._lock:
            n_flows = len(self.aggregator.flows)
        return {
            "running": self.running,
            "mode": self.mode,
            "source": self.source,
            "packets": self.stats["packets"],
            "flows": n_flows,
            "sim_t0": self.sim_t0,
            "elapsed": (time.time() - self.sim_t0) if self.sim_t0 else 0.0,
            "error": self.error,
        }


# ─── Windows over a flow snapshot ─────────────────────────────────────────────
def assign_windows(flows_df: pd.DataFrame, t0: float, window_size_s: float) -> pd.DataFrame:
    """
    Bucket flows into temporal windows anchored at ``t0`` (sim-time epoch).

    Returns a copy with an integer ``_window`` column (0 = first window).
    """
    if flows_df is None or flows_df.empty:
        return flows_df.copy()
    out = flows_df.copy()
    times = out["Timestamp"].astype("datetime64[s]").astype(np.int64)
    idx = np.floor((times - t0) / max(window_size_s, 1e-6)).astype(np.int64)
    out["_window"] = idx
    return out


def extract_window_states(
    flows_df: pd.DataFrame,
    t0: float,
    window_size_s: float,
    mapper: LiveStateMapper,
    scaler: OnlineScaler,
    start_window: int,
) -> Tuple[List[np.ndarray], int]:
    """
    Turn new temporal windows into standardized state vectors.

    Returns (states, next_window). ``start_window`` advances only when every
    intermediate window has at least one flow (so sparse windows are skipped
    instead of being padded with noise).
    """
    windowed = assign_windows(flows_df, t0, window_size_s)
    if windowed.empty:
        return [], start_window

    states: List[np.ndarray] = []
    windows = sorted(int(w) for w in windowed["_window"].unique())
    for w in windows:
        if w < start_window:
            continue
        group = windowed[windowed["_window"] == w]
        vec = mapper.state(group, scaler, window_size_s=window_size_s)
        states.append(vec)
        start_window = w + 1
    return states, start_window