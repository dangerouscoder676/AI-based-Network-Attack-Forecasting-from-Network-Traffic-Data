"""Tests for dashboard model-loading helper `_model_candidates` (non-Streamlit logic)."""

import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app"))

import session  # noqa: E402  (app/session.py — pure helpers we can test)


class TestModelCandidates(unittest.TestCase):
    def test_live_family_prefers_live_weights(self):
        root = tempfile.mkdtemp()
        with mock.patch("session.PROJECT_ROOT", root):
            cands = session._model_candidates("transformer", None, "live")
        # live-weighted checkpoints must come before the PCA-pipeline fallbacks
        self.assertLess(cands.index(os.path.join(root, "models", "best_live_model.pth")),
                        cands.index(os.path.join(root, "models", "transformer_model.pth")))

    def test_live_family_order(self):
        root = tempfile.mkdtemp()
        with mock.patch("session.PROJECT_ROOT", root):
            cands = session._model_candidates("transformer", None, "live")
        self.assertEqual(
            cands,
            [
                os.path.join(root, "models", "live_transformer_model.pth"),
                os.path.join(root, "models", "transformer_live_model.pth"),
                os.path.join(root, "models", "best_live_model.pth"),
                os.path.join(root, "models", "transformer_model.pth"),
                os.path.join(root, "models", "best_model.pth"),
            ],
        )

    def test_offline_family(self):
        root = tempfile.mkdtemp()
        with mock.patch("session.PROJECT_ROOT", root):
            cands = session._model_candidates("lstm", None, "offline")
        self.assertEqual(
            cands,
            [os.path.join(root, "models", "lstm_model.pth"),
             os.path.join(root, "models", "best_model.pth")],
        )


if __name__ == "__main__":
    unittest.main()