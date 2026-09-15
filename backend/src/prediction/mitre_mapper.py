from typing import List, Dict, Any

class MitreMapper:
    """MITRE ATT&CK stage mapping."""
    def __init__(self):
        self.stages = [
            {
                "name": "Benign",
                "description": "Normal network traffic without malicious intent.",
                "technique_ids": [],
                "tactic_id": "",
                "indicators": [],
                "mitigations": []
            },
            {
                "name": "Reconnaissance",
                "description": "Active scanning and network discovery.",
                "technique_ids": ["T1595", "T1046"],
                "tactic_id": "TA0043",
                "indicators": ["port scanning", "network discovery"],
                "mitigations": ["Restrict access to critical ports", "Implement network segmentation"]
            },
            {
                "name": "Initial Access",
                "description": "Exploiting vulnerabilities or brute-forcing access.",
                "technique_ids": ["T1190", "T1078", "T1133"],
                "tactic_id": "TA0001",
                "indicators": ["exploits", "brute force", "remote services"],
                "mitigations": ["Patch external-facing systems", "Enforce strong authentication"]
            },
            {
                "name": "Lateral Movement",
                "description": "Moving through the environment using remote services.",
                "technique_ids": ["T1021", "T1570", "T1550"],
                "tactic_id": "TA0008",
                "indicators": ["remote services", "lateral tool transfer"],
                "mitigations": ["Implement zero trust architecture", "Monitor internal lateral traffic"]
            },
            {
                "name": "Command & Control",
                "description": "Establishing communication with external malicious infrastructure.",
                "technique_ids": ["T1071", "T1573", "T1571"],
                "tactic_id": "TA0011",
                "indicators": ["app layer protocol", "encrypted channel"],
                "mitigations": ["Deploy NIDS for C2 detection", "Block known malicious domains"]
            },
            {
                "name": "Exfiltration",
                "description": "Stealing data from the compromised network.",
                "technique_ids": ["T1048", "T1041", "T1567"],
                "tactic_id": "TA0010",
                "indicators": ["exfil over alt protocol", "over C2"],
                "mitigations": ["Implement DLP solutions", "Monitor large outbound transfers"]
            }
        ]

    def get_stage_info(self, stage_idx: int) -> dict:
        if 0 <= stage_idx < len(self.stages):
            return self.stages[stage_idx]
        return self.stages[0]

    def get_stage_name(self, stage_idx: int) -> str:
        return self.get_stage_info(stage_idx)["name"]

    def get_all_stages(self) -> List[dict]:
        return self.stages

    def get_kill_chain_position(self, stage_idx: int) -> float:
        if stage_idx == 0:
            return 0.0
        return stage_idx / (len(self.stages) - 1)

    def map_label_to_stage(self, attack_label: str) -> int:
        label = str(attack_label).lower()
        if "benign" in label:
            return 0
        if any(x in label for x in ["portscan", "recon"]):
            return 1
        if any(x in label for x in ["bruteforce", "exploit"]):
            return 2
        if "infiltration" in label or "infilteration" in label:
            return 3
        if "bot" in label:
            return 4
        if "ddos" in label or "dos" in label:
            return 4
        return 2

    def get_indicators_for_stage(self, stage_idx: int) -> List[str]:
        return self.get_stage_info(stage_idx)["indicators"]

    def get_mitigations_for_stage(self, stage_idx: int) -> List[str]:
        return self.get_stage_info(stage_idx)["mitigations"]
