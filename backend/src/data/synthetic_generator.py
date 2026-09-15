import os
import numpy as np
from typing import Tuple, List

class SyntheticGenerator:
    """
    Generates synthetic temporal data for attack progressions.
    """
    def __init__(self):
        self.attack_stages = {
            'Normal': 0,
            'Reconnaissance': 1,
            'Initial Access': 2,
            'Lateral Movement': 3,
            'Command & Control': 4,
            'Exfiltration': 5
        }

    def generate(self, num_windows: int = 1000, num_features: int = 76) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Generate states and labels mimicking attack phases.
        """
        states = np.zeros((num_windows, num_features), dtype=np.float32)
        labels = np.zeros(num_windows, dtype=np.int64)
        
        # Generate phases
        # 70% Normal, 5% Recon, 8% Initial Access, 7% Lateral Movement, 5% C&C, 5% Exfil
        phases = [
            (0, int(num_windows * 0.70), 0),
            (int(num_windows * 0.70), int(num_windows * 0.75), 1),
            (int(num_windows * 0.75), int(num_windows * 0.83), 2),
            (int(num_windows * 0.83), int(num_windows * 0.90), 3),
            (int(num_windows * 0.90), int(num_windows * 0.95), 4),
            (int(num_windows * 0.95), num_windows, 5)
        ]
        
        for start, end, phase in phases:
            if start >= end:
                continue
                
            labels[start:end] = phase
            
            # Base features: Gaussian noise
            chunk = np.random.normal(loc=0.0, scale=1.0, size=(end - start, num_features))
            
            # Inject patterns based on phase
            if phase == 1: # Recon
                # High syn count (assume col 10), high dest port diversity (assume col -2)
                chunk[:, 10] = np.random.normal(loc=5.0, scale=1.0, size=(end - start))
                chunk[:, -2] = np.random.normal(loc=10.0, scale=2.0, size=(end - start))
                chunk[:, 2] = np.random.normal(loc=-2.0, scale=0.5, size=(end - start)) # low duration
                
            elif phase == 2: # Initial Access
                # High packet rates, brute force, high RST
                chunk[:, 11] = np.random.normal(loc=4.0, scale=1.0, size=(end - start)) # RST
                chunk[:, 3] = np.random.normal(loc=6.0, scale=1.5, size=(end - start)) # packets/s
                
            elif phase == 3: # Lateral Movement
                # Internal IPs, moderate duration
                chunk[:, 2] = np.random.normal(loc=1.0, scale=1.0, size=(end - start))
                
            elif phase == 4: # C&C
                # Periodic beaconing (regular IAT -> low variance)
                chunk[:, 12] = np.random.normal(loc=-3.0, scale=0.2, size=(end - start)) # IAT std
                chunk[:, 4] = np.random.normal(loc=-2.0, scale=0.5, size=(end - start)) # small packets
                
            elif phase == 5: # Exfiltration
                # High fwd bytes, long duration
                chunk[:, 5] = np.random.normal(loc=8.0, scale=2.0, size=(end - start)) # fwd bytes
                chunk[:, 2] = np.random.normal(loc=5.0, scale=1.0, size=(end - start)) # high duration
                
            states[start:end] = chunk
            
        feature_names = [f"feature_{i}" for i in range(num_features)]
        return states, labels, feature_names

    def generate_sequences(self, context_length: int = 16) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate contextual sequences with explicit next-state targets.

        Returns:
            sequences:     [num_seq, context_length, num_features]
            next_states:   [num_seq, num_features]
            seq_labels:    [num_seq] binary infiltration label of predicted state
            seq_stages:    [num_seq] ATT&CK stage of predicted state
        """
        states, labels, _ = self.generate(num_windows=1000, num_features=76)

        num_sequences = len(states) - context_length
        if num_sequences <= 0:
            return (np.zeros((0, context_length, 76), dtype=np.float32),
                    np.zeros((0, 76), dtype=np.float32),
                    np.zeros(0, dtype=np.int64),
                    np.zeros(0, dtype=np.int64))

        sequences = np.zeros((num_sequences, context_length, states.shape[1]), dtype=np.float32)
        next_states = np.zeros((num_sequences, states.shape[1]), dtype=np.float32)
        seq_labels = np.zeros(num_sequences, dtype=np.int64)
        seq_stages = np.zeros(num_sequences, dtype=np.int64)

        for i in range(num_sequences):
            sequences[i] = states[i:i + context_length]
            next_states[i] = states[i + context_length]
            seq_labels[i] = 1 if labels[i + context_length] > 0 else 0
            seq_stages[i] = labels[i + context_length]

        return sequences, next_states, seq_labels, seq_stages

    def save(self, output_dir: str):
        """Save to npz with sequences, next_states, infiltration labels and stages."""
        os.makedirs(output_dir, exist_ok=True)
        sequences, next_states, seq_labels, seq_stages = self.generate_sequences()

        np.savez_compressed(
            os.path.join(output_dir, "synthetic_data.npz"),
            sequences=sequences,
            next_states=next_states,
            labels=seq_labels,
            stages=seq_stages
        )
