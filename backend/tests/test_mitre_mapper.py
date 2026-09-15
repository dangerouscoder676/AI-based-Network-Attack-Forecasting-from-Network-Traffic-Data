import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.prediction.mitre_mapper import MitreMapper

class TestMitreMapper(unittest.TestCase):
    def setUp(self):
        self.mapper = MitreMapper()

    def test_get_stage_info(self):
        info = self.mapper.get_stage_info(0)
        self.assertEqual(info['name'], "Benign")
        info_recon = self.mapper.get_stage_info(1)
        self.assertEqual(info_recon['name'], "Reconnaissance")
        self.assertIn("T1595", info_recon['technique_ids'])

    def test_map_label_to_stage(self):
        self.assertEqual(self.mapper.map_label_to_stage("Benign"), 0)
        self.assertEqual(self.mapper.map_label_to_stage("FTP-BruteForce"), 2)
        self.assertEqual(self.mapper.map_label_to_stage("Infilteration"), 3)
        self.assertEqual(self.mapper.map_label_to_stage("Bot"), 4)

    def test_get_kill_chain_position(self):
        self.assertEqual(self.mapper.get_kill_chain_position(0), 0.0)
        self.assertEqual(self.mapper.get_kill_chain_position(5), 1.0)
        self.assertGreater(self.mapper.get_kill_chain_position(3), self.mapper.get_kill_chain_position(1))

if __name__ == '__main__':
    unittest.main()
