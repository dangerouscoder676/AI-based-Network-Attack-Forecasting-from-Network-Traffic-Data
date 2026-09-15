import os
import sys
import unittest

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT,):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.prediction.attack_simulator import (
    build_phase_plan,
    build_story,
    describe_scan,
    fix_howtos,
    forecast_from_scan,
    quiet_network_forecast,
    simulate_attack,
)


def _live_model():
    """Load the live-trained transformer without importing the `app` package
    (AppTest may have registered an `app` module in sys.modules, which shadows
    the directory in the same test process)."""
    from session import load_model
    return load_model("transformer", family="live")


class TestBuildPhasePlan(unittest.TestCase):
    def test_default_plan(self):
        plan = build_phase_plan(n_benign=20)
        self.assertEqual(len(plan), 20 + 4 + 4 + 4)
        self.assertTrue(all(p == 0 for p in plan[:20]))
        self.assertEqual(plan[20], 1)
        self.assertEqual(plan[24], 2)

    def test_custom_attacks(self):
        plan = build_phase_plan(n_benign=5, attacks=[(1, 2), (3, 1)])
        self.assertEqual(len(plan), 5 + 2 + 1 + 4)
        self.assertEqual(plan[5:7], [1, 1])
        self.assertEqual(plan[7], 3)
        self.assertTrue(all(p == 0 for p in plan[-4:]))


class DescribeScanTests(unittest.TestCase):
    def _summary(self, **kw):
        base = {"hosts": 8, "open_ports": 3, "exposed_hosts": 2, "risk_high": 0,
                "risk_score": 6.0, "level": "MEDIUM", "mitre": ["T1046 Network Service Discovery"]}
        base.update(kw)
        return base

    def test_plain_headline(self):
        out = describe_scan(pd.DataFrame(), self._summary(), {})
        self.assertIn("8", out["headline"])
        self.assertIn("2", out["headline"])
        self.assertEqual(out["level"], "MEDIUM")
        self.assertIn("AI", out["ai_sentence"])

    def test_recommendations_present_when_exposed(self):
        out = describe_scan(pd.DataFrame(), self._summary(), {})
        self.assertTrue(len(out["recommendations"]) >= 3)

    def test_calm_network(self):
        out = describe_scan(pd.DataFrame(), self._summary(open_ports=0, exposed_hosts=0,
                                                          level="NONE", risk_score=0.0), {})
        self.assertIn("Nothing urgent", out["recommendations"][0])

    def test_safe_network_reassures_not_alarms(self):
        out = describe_scan(pd.DataFrame(),
                            self._summary(level="LOW", risk_score=2.0, open_ports=4,
                                          exposed_hosts=2),
                            {"peak": 0.95, "stage_display": "Initial Access"})
        self.assertTrue(out["safe"])
        self.assertTrue(out["verdict"]["safe"])
        self.assertIn("looks safe", out["verdict"]["title"])
        # A safe network must never surface an AI percentage.
        self.assertNotIn("100%", out["ai_sentence"])
        self.assertNotIn("0.95", out["ai_sentence"])
        self.assertNotIn("an attacker's first move", out["ai_sentence"])

    def test_exposed_network_alerts(self):
        out = describe_scan(pd.DataFrame(),
                            self._summary(level="HIGH", risk_score=12.0, open_ports=6,
                                          exposed_hosts=3, risk_high=2),
                            {"peak": 0.9, "stage_display": "Lateral Movement"})
        self.assertFalse(out["safe"])
        self.assertFalse(out["verdict"]["safe"])
        self.assertIn("an attacker's first move", out["ai_sentence"])

    def test_forecast_sentence(self):
        forecast = {"peak": 0.87, "stage_display": "Lateral Movement"}
        out = describe_scan(pd.DataFrame(), self._summary(), forecast)
        self.assertIn("87%", out["ai_sentence"])
        self.assertIn("Lateral Movement", out["ai_sentence"])


class BuildStoryTests(unittest.TestCase):
    def _timeline(self, phases=(0, 0, 1, 2)):
        return [{"phase_name": "normal" if p == 0 else f"phase{p}", "phase": p,
                 "prob": 0.5, "stage": p, "stage_name": "Benign"} for p in phases]

    def test_alerting_story_at_high_peak(self):
        story = build_story(self._timeline(), 0.95, "Exfiltration")
        self.assertTrue(any("95%" in s for s in story))
        self.assertTrue(any("intrusion" in s for s in story))

    def test_low_peak_story(self):
        story = build_story(self._timeline(), 0.1, "Benign")
        self.assertTrue(any("0%".replace("0%", "10%") in s or "10%" in s for s in story))


class FixHowtoTests(unittest.TestCase):
    def _df(self):
        return pd.DataFrame([
            {"ip": "192.168.1.5", "vendor": "", "os_guess": "Windows", "services": "445,3389"},
            {"ip": "192.168.1.30", "vendor": "Raspberry Pi", "os_guess": "Linux", "services": "22,9999"},
            {"ip": "192.168.1.99", "vendor": "Xiaomi", "os_guess": "", "services": ""},
        ])

    def test_groups_doors_per_device(self):
        out = fix_howtos(self._df())
        self.assertEqual(len(out["devices"]), 2)
        dev = out["devices"][0]
        self.assertEqual(dev["ip"], "192.168.1.5")
        self.assertEqual([d["port"] for d in dev["doors"]], [445, 3389])
        # Known services carry a real name + close-guide; unknown ports a fallback.
        self.assertEqual(dev["doors"][0]["name"], "Windows file sharing (SMB)")
        self.assertTrue(dev["doors"][0]["how"])
        self.assertEqual(out["devices"][1]["doors"][1]["name"],
                         "A program listening on this port")

    def test_no_doors_yields_nothing(self):
        df = pd.DataFrame([{"ip": "10.0.0.2", "vendor": "Xiaomi", "os_guess": "",
                            "services": ""}])
        self.assertEqual(fix_howtos(df)["devices"], [])


class QuietForecastTests(unittest.TestCase):
    def test_benign_replay_stays_quiet(self):
        model = _live_model()
        if model is None:
            self.skipTest("live model weights not present")
        out = quiet_network_forecast(model, n_windows=20, seed=7)
        self.assertIn("peak", out)
        self.assertLess(out["peak"], 0.3, "routine traffic should read quiet")


class SimulateAttackTests(unittest.TestCase):
    def test_attack_against_live_model(self):
        model = _live_model()
        if model is None:
            self.skipTest("live model weights not present")

        phases = build_phase_plan(n_benign=16, attacks=[(1, 5), (2, 5)])
        out = simulate_attack(model, phases, seed=7)
        self.assertNotIn("error", out)
        self.assertEqual(len(out["timeline"]), len(phases))
        self.assertTrue(len(out["story"]) >= 2)

        benign_avg = np.mean([t["prob"] for t in out["timeline"][:8]])
        attack_max = max(t["prob"] for t in out["timeline"][16:26])
        # The live model must hear the attack clearly.
        self.assertLess(benign_avg, 0.15, "benign window scored as an attack")
        self.assertGreater(attack_max, 0.7, "attack windows scored as benign")


class ScanForecastTests(unittest.TestCase):
    def test_forecast_from_scan(self):
        model = _live_model()
        if model is None:
            self.skipTest("live model weights not present")

        df = pd.DataFrame([
            {"ip": "192.168.1.10", "vendor": "Apple", "os_guess": "Unknown",
             "n_open_ports": 1, "open_ports": "22", "services": "22",
             "risk_score": 3.0, "risk_label": "LOW", "ttl": 64, "mac": "00:11:22:33:44:55"},
            {"ip": "192.168.1.20", "vendor": "Raspberry Pi", "os_guess": "Linux",
             "n_open_ports": 2, "open_ports": "80,22", "services": "80,22",
             "risk_score": 9.0, "risk_label": "MEDIUM", "ttl": 64, "mac": "b8:27:eb:00:00:01"},
        ])
        out = forecast_from_scan(df, probed_ports=[22, 80], model=model,
                                 ctx_win=12, attack_win=1, k_steps=8, seed=42)
        self.assertNotIn("error", out)
        self.assertEqual(len(out["probabilities"]), 8)
        self.assertGreaterEqual(out["peak"], 0.5, "scan recon must be flagged")


if __name__ == "__main__":
    unittest.main()
