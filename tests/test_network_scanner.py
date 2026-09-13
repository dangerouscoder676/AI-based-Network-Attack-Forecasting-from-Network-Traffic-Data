"""Tests for the active Network Scanner (host discovery, port scan, risk scoring).

No real packets are sent — discovery is mocked and connect-scanning is replaced
with a fake socket so the suite runs anywhere, offline, without Npcap.
"""

import socket
import sys
import os
import unittest
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.network_scanner import (
    DiscoveredHost,
    _ttl_os_guess,
    _lookup_vendor,
    emit_probe_flows,
    host_risk,
    network_risk_summary,
    scan_ports,
)
from src.data.network_scanner import icmp_discovery, guess_local_network


class TestTtlGuess(unittest.TestCase):
    def test_windows_linux_device(self):
        self.assertEqual(_ttl_os_guess(64), "Linux / *nix")
        self.assertEqual(_ttl_os_guess(128), "Windows")
        self.assertEqual(_ttl_os_guess(255), "Network device (router/switch)")
        self.assertEqual(_ttl_os_guess(None), "unknown")
        self.assertEqual(_ttl_os_guess(120), "Windows")


class TestVendorLookup(unittest.TestCase):
    def test_known_oui(self):
        self.assertEqual(_lookup_vendor("dc:a6:32:ab:cd:ef"), "Raspberry Pi")
        self.assertEqual(_lookup_vendor("b8:27:eb:12:34:56"), "Raspberry Pi")

    def test_unknown_oui(self):
        self.assertEqual(_lookup_vendor("00:11:22:33:44:55"), "")


class TestHostRisk(unittest.TestCase):
    def _hosts(self):
        return [
            DiscoveredHost(ip="192.168.1.10", mac="dc:a6:32:aa:bb:cc", open_ports=[22, 80, 443]),
            DiscoveredHost(ip="192.168.1.11", open_ports=[3389, 445]),
            DiscoveredHost(ip="192.168.1.12", open_ports=[53]),
            DiscoveredHost(ip="192.168.1.13", open_ports=[]),
        ]

    def test_scoring_and_labels(self):
        df = host_risk(self._hosts())
        self.assertEqual(len(df), 4)
        self.assertEqual(list(df["risk_label"]), ["MEDIUM", "MEDIUM", "LOW", "NONE"])
        self.assertEqual(set(df["mitre_techniques"].tolist()), {"", "T1046", "T1190;T1046"})
        self.assertEqual(df.loc[df["ip"] == "192.168.1.10", "services"].iloc[0], "ssh,http,https")

    def test_management_ports_flagged_as_targets(self):
        df = host_risk(self._hosts())
        self.assertTrue(df.loc[df["ip"] == "192.168.1.11", "likely_target"].iloc[0])
        self.assertFalse(df.loc[df["ip"] == "192.168.1.12", "likely_target"].iloc[0])

    def test_summary(self):
        df = host_risk(self._hosts())
        s = network_risk_summary(df, 254)
        self.assertEqual(s["hosts"], 4)
        self.assertEqual(s["exposed_hosts"], 3)
        self.assertIn("T1046", " ".join(s["mitre"]))


class TestScanPorts(unittest.TestCase):
    def test_scan_with_fake_connect(self):
        ok_socket = mock.MagicMock()

        def fake_connect(addr, timeout):
            ip, port = addr
            if ip == "192.168.1.10" and port in (22, 80, 443):
                return ok_socket
            raise OSError("refused")

        host = DiscoveredHost(ip="192.168.1.10")
        with mock.patch.object(socket, "create_connection", side_effect=fake_connect):
            scan_ports([host], ports=[22, 80, 443, 444], workers=4, timeout_s=0.1)
        self.assertEqual(host.open_ports, [22, 80, 443])


class TestDiscovery(unittest.TestCase):
    def test_icmp_uses_subprocess(self):
        net = "10.99.1.2/31"  # exactly 2 addresses -> fast
        with mock.patch("subprocess.run") as run:
            run.return_value = mock.Mock(
                stdout="Reply from 10.99.1.2: bytes=32 time=2ms TTL=128\n",
                returncode=0,
            )
            hosts = icmp_discovery(net, workers=2, timeout_s=1)
        self.assertEqual(sorted(h.ip for h in hosts), ["10.99.1.2", "10.99.1.3"])
        self.assertTrue(all(h.os_guess == "Windows" for h in hosts))
        self.assertTrue(run.called)


class TestGuessLocalNetwork(unittest.TestCase):
    def test_default_network_shape(self):
        with mock.patch("socket.socket"):
            net = guess_local_network("192.168.5.100")
        self.assertEqual(net, "192.168.5.0/24")


class TestEmitProbeFlows(unittest.TestCase):
    def test_benign_windows_stay_distinct(self):
        """Regression: benign windows must NOT merge into one mega-flow.

        The flow aggregator never expires flows, so identical 5-tuples across
        windows collapse into a single flow and the world model ends up with a
        degenerate 1-2 window context. Each benign window must use a distinct
        destination so per-window states are preserved.
        """
        from src.data.pcap_parser import FlowAggregator
        from src.data.live_capture import LiveStateMapper, OnlineScaler, extract_window_states, STATE_DIM

        df = host_risk([DiscoveredHost(ip="192.168.29.20", open_ports=[22, 80])])
        agg = FlowAggregator()
        emit_probe_flows(df, agg, src_ip="192.168.29.5", benign_before=12, seed=7,
                         window_size_s=5.0, t0=1_700_000_000.0, attack_windows=1)
        states, _ = extract_window_states(agg.snapshot(), 1_700_000_000.0, 5.0,
                                          LiveStateMapper(), OnlineScaler(STATE_DIM), 0)
        # 12 benign + 1 attack window
        self.assertGreaterEqual(len(states), 13)
        self.assertLessEqual(len(states), 14)
        self.assertEqual(len(states[0]), STATE_DIM)

    def test_probes_replayed_to_every_probed_port(self):
        from src.data.pcap_parser import FlowAggregator

        df = host_risk([DiscoveredHost(ip="192.168.29.20", open_ports=[22])])
        agg = FlowAggregator()
        n = emit_probe_flows(df, agg, src_ip="192.168.29.5", benign_before=0, seed=7,
                             window_size_s=5.0, t0=1_700_000_000.0,
                             attack_windows=1, probed_ports=[22, 80, 443])
        # SYN probes + replies for all 3 probed ports
        self.assertGreaterEqual(n, 3 * 2)


if __name__ == "__main__":
    unittest.main()