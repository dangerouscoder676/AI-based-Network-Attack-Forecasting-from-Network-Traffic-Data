import os
import unittest

import streamlit.testing.v1 as stt

APP_MAIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "app", "app.py")


@unittest.skipUnless(
    os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "models", "best_live_model.pth")),
    "live model weights absent — run scripts/train_live.py first")
class OneClickConsoleTests(unittest.TestCase):
    def test_console_boots_with_three_tabs(self):
        at = stt.AppTest.from_file(APP_MAIN, default_timeout=120).run()
        self.assertEqual(at.exception, [])
        self.assertEqual(len(at.tabs), 3)

    def test_simulated_attack_plays(self):
        at = stt.AppTest.from_file(APP_MAIN, default_timeout=240).run()
        at.tabs[1].button[0].click()
        at.run()
        self.assertEqual(at.exception, [])
        self.assertIn("one_sim", at.session_state)
        sim = at.session_state["one_sim"]
        self.assertNotIn("error", sim)
        self.assertGreater(len(sim["timeline"]), 30)
        self.assertGreater(sim["peak"], 0.5)  # the AI must hear the attack
        joined = " | ".join(m.value for m in at.markdown)
        self.assertIn("What just happened", joined)


if __name__ == "__main__":
    unittest.main()