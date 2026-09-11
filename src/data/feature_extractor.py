import pandas as pd
import numpy as np
from typing import List

class FeatureExtractor:
    """
    Extracts and computes derived features from flow-level DataFrames.
    """
    def __init__(self, feature_names: List[str] = None):
        if feature_names is None:
            # Fallback if config is not passed directly
            self.base_features = []
        else:
            self.base_features = feature_names

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract base features and compute derived features.
        """
        if df.empty:
            return df
            
        out_df = df.copy()
        
        # Ensure all base features exist
        for f in self.base_features:
            if f not in out_df.columns:
                out_df[f] = 0.0
                
        # Aliases for robust derived features
        # Support both PCAP output names and CIC-IDS CSV names
        total_packets = out_df.get('total_packets', out_df.get('Tot Fwd Pkts', 0) + out_df.get('Tot Bwd Pkts', 0))
        total_bytes = out_df.get('total_bytes', out_df.get('TotLen Fwd Pkts', 0) + out_df.get('TotLen Bwd Pkts', 0))
        duration = out_df.get('Flow Duration', 0) / 1e6 # micro to seconds
        
        syn_cnt = out_df.get('SYN Flag Cnt', out_df.get('SYN_Flag_Cnt', 0))
        ack_cnt = out_df.get('ACK Flag Cnt', out_df.get('ACK_Flag_Cnt', 0))
        rst_cnt = out_df.get('RST Flag Cnt', out_df.get('RST_Flag_Cnt', 0))
        
        fwd_pkts = out_df.get('Tot Fwd Pkts', 0)
        bwd_pkts = out_df.get('Tot Bwd Pkts', 0)
        fwd_bytes = out_df.get('TotLen Fwd Pkts', 0)
        bwd_bytes = out_df.get('TotLen Bwd Pkts', 0)

        # Compute derived features
        out_df['syn_to_total_ratio'] = syn_cnt / (total_packets + 1e-8)
        out_df['ack_to_total_ratio'] = ack_cnt / (total_packets + 1e-8)
        out_df['rst_to_total_ratio'] = rst_cnt / (total_packets + 1e-8)
        out_df['bytes_per_packet'] = total_bytes / (total_packets + 1e-8)
        out_df['fwd_bwd_packet_ratio'] = fwd_pkts / (bwd_pkts + 1e-8)
        out_df['fwd_bwd_byte_ratio'] = fwd_bytes / (bwd_bytes + 1e-8)
        
        # port diversity is 0 for single flows, will be computed per-window later, but added as placeholder
        out_df['port_diversity'] = 0.0
        out_df['flow_rate'] = total_packets / (duration + 1e-8)
        
        # Clean inf/nans that might result from division by very small numbers
        out_df = out_df.replace([np.inf, -np.inf], np.nan)
        out_df = out_df.fillna(0.0)
        
        # Select the desired columns
        all_features = self.get_feature_names()
        
        # Keep non-feature columns that are important (like Timestamp, label_cat)
        keep_cols = [c for c in ['Timestamp', 'label_cat', 'Src IP', 'Dst IP', 'Dst Port', 'Src Port'] if c in out_df.columns]
        
        # Ensure all requested features exist before filtering
        for f in all_features:
            if f not in out_df.columns:
                out_df[f] = 0.0
                
        return out_df[keep_cols + all_features]

    def get_feature_names(self) -> List[str]:
        """Get list of all extracted feature names."""
        derived = [
            'syn_to_total_ratio',
            'ack_to_total_ratio',
            'rst_to_total_ratio',
            'bytes_per_packet',
            'fwd_bwd_packet_ratio',
            'fwd_bwd_byte_ratio',
            'port_diversity',
            'flow_rate'
        ]
        return self.base_features + derived
