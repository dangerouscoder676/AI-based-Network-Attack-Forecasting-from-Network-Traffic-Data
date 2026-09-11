import numpy as np
import pandas as pd
from typing import Tuple, List

class StateBuilder:
    """
    Constructs temporal state vectors from flow data.
    """
    def __init__(self):
        # Maps MITRE ATT&CK stages
        self.attack_stage_mapping = {
            0: 0, # Benign
            1: 2, # Brute Force -> Initial Access
            2: 2, # DoS -> Initial Access
            3: 2, # DDoS -> Initial Access
            4: 2, # Web Attack -> Initial Access
            5: 3, # Infiltration -> Lateral Movement
            6: 4  # Botnet -> Command & Control
        }

    def build_states(self, df: pd.DataFrame, window_size: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """
        Groups flows into temporal windows and aggregates.
        """
        if df.empty or 'Timestamp' not in df.columns:
            return np.array([]), np.array([])
            
        df = df.sort_values('Timestamp').reset_index(drop=True)
        
        # Start of first window
        start_time = df['Timestamp'].iloc[0]
        
        # Create window index
        df['window_idx'] = ((df['Timestamp'] - start_time).dt.total_seconds() // window_size).astype(int)
        
        states = []
        labels = []
        
        # Numerical features to aggregate
        num_features = [c for c in df.columns if c not in ['Timestamp', 'label_cat', 'Src IP', 'Dst IP', 'Dst Port', 'Src Port', 'Protocol', 'window_idx'] and not c.startswith('proto_')]
        
        for idx, group in df.groupby('window_idx'):
            # Aggregate stats
            means = group[num_features].mean().values
            stds = group[num_features].std().fillna(0).values
            maxes = group[num_features].max().values
            
            # Additional window-level features
            total_flows = len(group)
            
            # Port diversity
            if 'Dst Port' in group.columns:
                unique_dst_ports = group['Dst Port'].nunique()
            else:
                unique_dst_ports = 1
                
            if 'Src IP' in group.columns:
                unique_src_ips = group['Src IP'].nunique()
            else:
                unique_src_ips = 1
                
            # Most common label (map to MITRE stage if possible)
            if 'label_cat' in group.columns:
                window_label_raw = group['label_cat'].mode().iloc[0] if not group['label_cat'].empty else 0
                window_label = self.attack_stage_mapping.get(window_label_raw, 0)
            else:
                window_label = 0
                
            state_vec = np.concatenate([means, stds, maxes, [total_flows, unique_dst_ports, unique_src_ips]])
            
            # We override port diversity for states (assuming it's a feature, if not it gets appended as dimension)
            # Actually just returning the concatenated vector is robust.
            
            states.append(state_vec)
            labels.append(window_label)
            
        return np.array(states), np.array(labels)

    def build_sequences(self, states: np.ndarray, labels: np.ndarray, context_length: int = 16) -> Tuple[np.ndarray, np.ndarray]:
        """
        Construct sliding-window sequences.
        """
        if len(states) < context_length:
            return np.array([]), np.array([])
            
        num_sequences = len(states) - context_length + 1
        
        sequences = np.zeros((num_sequences, context_length, states.shape[1]), dtype=np.float32)
        seq_labels = np.zeros(num_sequences, dtype=np.int64)
        
        for i in range(num_sequences):
            sequences[i] = states[i:i+context_length]
            # Label is the label of the last window in the context
            seq_labels[i] = labels[i+context_length-1]
            
        return sequences, seq_labels
