import os
import glob
import numpy as np
import pandas as pd
from typing import List, Dict

class CICIDSLoader:
    """
    Loader for CIC-IDS-2018 CSV datasets.
    """
    def __init__(self):
        self.label_mapping = {
            'Benign': 0,
            'FTP-BruteForce': 1,
            'SSH-Bruteforce': 1,
            'DoS attacks-GoldenEye': 2,
            'DoS attacks-Slowloris': 2,
            'DoS attacks-SlowHTTPTest': 2,
            'DoS attacks-Hulk': 2,
            'DDoS attacks-LOIC-HTTP': 3,
            'DDoS attack-LOIC-UDP': 3,
            'DDoS attack-HOIC': 3,
            'Brute Force -Web': 4,
            'Brute Force -XSS': 4,
            'SQL Injection': 4,
            'Infilteration': 5,
            'Bot': 6
        }

    def _clean_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names by stripping whitespace."""
        df.columns = [col.strip() for col in df.columns]
        return df

    def _map_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map raw string labels to category integers."""
        if 'Label' in df.columns:
            # Map known labels, unmapped become -1 or could be dropped.
            # Using get to safely handle unknowns (falling back to benign or unknown)
            df['label_cat'] = df['Label'].map(lambda x: self.label_mapping.get(x, 0))
        else:
            df['label_cat'] = 0
        return df

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove inf/nan, drop duplicates."""
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        df = df.drop_duplicates()
        return df
        
    def _parse_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse Timestamp column and sort."""
        if 'Timestamp' in df.columns:
            # Errors coerce to handle malformed dates if any
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce', format='mixed')
            df = df.dropna(subset=['Timestamp'])
            df = df.sort_values('Timestamp').reset_index(drop=True)
        return df

    def load_single_file(self, filepath: str) -> pd.DataFrame:
        """
        Load, clean, and map a single CSV file.
        """
        # Read dataset. Using low_memory=False to avoid mixed type warnings
        try:
            df = pd.read_csv(filepath, low_memory=False)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return pd.DataFrame()
            
        df = self._clean_columns(df)
        df = self._clean_data(df)
        df = self._parse_time(df)
        df = self._map_labels(df)
        return df

    def load(self, data_dir: str) -> pd.DataFrame:
        """
        Load all CSV files in the directory and combine them.
        """
        csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
        if not csv_files:
            print(f"No CSV files found in {data_dir}")
            return pd.DataFrame()
            
        dfs = []
        for file in csv_files:
            print(f"Loading {file}...")
            df = self.load_single_file(file)
            if not df.empty:
                dfs.append(df)
                
        if not dfs:
            return pd.DataFrame()
            
        combined_df = pd.concat(dfs, ignore_index=True)
        # Final sort just in case files overlap in time
        if 'Timestamp' in combined_df.columns:
            combined_df = combined_df.sort_values('Timestamp').reset_index(drop=True)
            
        return combined_df
