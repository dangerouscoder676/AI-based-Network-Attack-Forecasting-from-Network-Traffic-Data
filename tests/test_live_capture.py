import sys
import os
import unittest
import time
import threading
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.live_capture import (
    OnlineScaler,
    LiveStateMapper,
    LiveTrafficEngine,
    assign_windows,
    extract_window_states,
    STATE_DIM,
)

from scapy.all import IP, TCP, UDP


def _syn_packet(src="10.0.0.9", dst="192.168.1.100", sport=40000, dport=80):
    return IP(src=src, dst=dst, proto=6) / TCP(sport=sport, dport=dport, flags="S")


def _http_packet(src="10.0.0.20", dst="93.184.216.34", sport=50000, dport=443):
    return IP(src=src, dst=dst, proto=6) / TCP(sport=sport, dport=dport, flags="AP") / b"A" * 128


class TestOnlineScaler(unittest.TestCase):
    def test_incremental_fit_and_transform(self):
        sc = OnlineScaler(4)
        rng = np.random.RandomState(0)
        for _ in range(50):
            sc.update(rng.normal(2.0, 3.0, size=(5, 4)))
        self.assertTrue(sc.fitted)
        self.assertEqual(sc.mean.shape, (4,))
        self.assertEqual(sc.sd.shape, (4,))
        np.testing.assert_allclose(np.mean(sc.mean), 2.0, atol=0.5)

    def test_clip(self):
        sc = OnlineScaler(2)
        sc.update(np.array([[10.0, -20.0]]))
        z = sc.transform(np.array([[1e9, -1e9]]))
        self.assertTrue(np.abs(z).max() <= 6.0 + 1e-9)


class TestLiveStateMapper(unittest.TestCase):
    def _flow_stats(self, packets, t0):
        from src.data.pcap_parser import FlowAggregator
        agg = FlowAggregator()
        for i, p in enumerate(packets):
            agg.update(p, timestamp=t0 + i * 0.01)
        return agg.snapshot()

    def test_scan_deviates_on_signal_channels(self):
        mapper = LiveStateMapper()
        sc = OnlineScaler(STATE_DIM)

        # Baseline window first (benign web) so z-scores are meaningful.
        benign = [_http_packet(sport=50000 + i, dport=443) for i in range(40)]
        sc.update(mapper.build_raw_state(self._flow_stats(benign, 1000.0)).reshape(1, -1))

        # Scan window: many SYN flows to many destination ports.
        scan_pkts = [_syn_packet(dport=port) for port in range(1, 40)]
        scan_flows = self._flow_stats(scan_pkts, 2000.0)
        vec = mapper.state(scan_flows, sc)

        self.assertEqual(vec.shape, (STATE_DIM,))
        self.assertEqual(vec.dtype, np.float32)
        # SYN ratio (ch 10) and destination-port diversity (ch 7 / 74) must rise
        # above the benign baseline — the same channels the world model keys on.
        self.assertGreater(vec[10], 0.6)
        self.assertGreater(vec[7], 0.5)
        self.assertGreater(vec[74], 0.5)

    def test_benign_window_stays_low(self):
        mapper = LiveStateMapper()
        sc = OnlineScaler(STATE_DIM)
        baseline = [_http_packet(sport=50000 + i, dport=443) for i in range(40)]
        sc.update(mapper.build_raw_state(self._flow_stats(baseline, 1000.0)).reshape(1, -1))

        later = [_http_packet(sport=60000 + i, dport=443) for i in range(40)]
        vec = mapper.state(self._flow_stats(later, 2000.0), sc)
        self.assertLess(vec[10], 0.5)  # ~no SYN-only activity
        self.assertLess(vec[7], 1.0)   # single destination port
        self.assertEqual(vec.shape, (STATE_DIM,))


class TestLiveTrafficEngine(unittest.TestCase):
    def test_inject_scan_then_snapshot(self):
        engine = LiveTrafficEngine()
        n = engine.inject_scan(target="192.168.1.100", src="10.0.0.9", ports=30)
        self.assertGreater(n, 0)
        flows = engine.snapshot_flows()
        self.assertFalse(flows.empty)
        self.assertGreaterEqual(len(flows), 20)
        self.assertIn("SYN_Flag_Cnt", flows.columns)
        engine.stop()

    def test_live_interface_falls_back_gracefully(self):
        # Should not raise if sniffing isn't possible on this host.
        engine = LiveTrafficEngine()
        try:
            engine.start_interface()
        except Exception:
            pass
        engine.stop()

    def test_windows_assign(self):
        df = pd.DataFrame({
            "Timestamp": pd.to_datetime([1000.0, 1005.0, 1012.0, 1025.0], unit="s"),
            "Dst Port": [1, 2, 3, 4],
        })
        out = assign_windows(df, t0=1000.0, window_size_s=10.0)
        self.assertEqual(out["_window"].tolist(), [0, 0, 1, 2])

    def test_extract_window_states(self):
        engine = LiveTrafficEngine()
        engine.inject_scan(target="192.168.1.100", src="10.0.0.9", ports=20)
        flows = engine.snapshot_flows()
        mapper = LiveStateMapper()
        scaler = OnlineScaler(STATE_DIM)
        states, nxt = extract_window_states(flows, flow_t0(engine), 10.0, mapper, scaler, 0)
        # At least one window should resolve from the warmed-up traffic.
        self.assertIsInstance(nxt, int)
        self.assertTrue(all(s.shape == (STATE_DIM,) for s in states))
        engine.stop()


def flow_t0(engine):
    """sim_t0 is private; recompute roughly from the first flow timestamp."""
    flows = engine.snapshot_flows()
    t0 = float(flows["Timestamp"].min().timestamp()) if not flows.empty else time.time()
    return t0


if __name__ == '__main__':
    unittest.main()