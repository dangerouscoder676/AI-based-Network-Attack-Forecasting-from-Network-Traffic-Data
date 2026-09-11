import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

class FeatureNormalizer:
    """
    Normalizer applying log transforms and RobustScaling.
    """
    def __init__(self):
        self.scaler = RobustScaler()
        # Features that benefit from log1p transform
        self.log_keywords = ['Byts', 'Pkts', 'Duration', 'IAT', 'Len', 'bytes', 'packets', 'rate']
        self.log_features = []
        self.fitted = False
        self.feature_columns = []

    def _get_log_features(self, columns):
        """Identify which features to log-transform."""
        return [c for c in columns if any(k.lower() in c.lower() for k in self.log_keywords)]

    def fit(self, df: pd.DataFrame):
        """Fit scaler to the data."""
        self.feature_columns = [c for c in df.columns if c not in ['Timestamp', 'label_cat', 'Src IP', 'Dst IP', 'Dst Port', 'Src Port', 'Protocol']]
        self.log_features = self._get_log_features(self.feature_columns)
        
        # Temporarily apply log to fit scaler
        temp_df = df[self.feature_columns].copy()
        for col in self.log_features:
            # Handle potential negative values (like min values) by clipping
            temp_df[col] = np.log1p(np.clip(temp_df[col], a_min=0, a_max=None))
            
        self.scaler.fit(temp_df)
        self.fitted = True

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform the dataframe using fitted scaler."""
        if not self.fitted:
            raise ValueError("Normalizer must be fitted before calling transform.")
            
        out_df = df.copy()
        
        # Apply log1p
        for col in self.log_features:
            if col in out_df.columns:
                out_df[col] = np.log1p(np.clip(out_df[col], a_min=0, a_max=None))
                
        # Handle Protocol one-hot encoding if present
        if 'Protocol' in out_df.columns:
            protos = pd.get_dummies(out_df['Protocol'], prefix='proto')
            out_df = pd.concat([out_df, protos], axis=1)
            
        # Scale numerical features
        out_df[self.feature_columns] = self.scaler.transform(out_df[self.feature_columns])
        
        return out_df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(df)
        return self.transform(df)

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform data."""
        if not self.fitted:
            raise ValueError("Normalizer must be fitted before calling inverse_transform.")
            
        out_df = df.copy()
        out_df[self.feature_columns] = self.scaler.inverse_transform(out_df[self.feature_columns])
        
        for col in self.log_features:
            if col in out_df.columns:
                out_df[col] = np.expm1(out_df[col])
                
        return out_df

    def save(self, path: str):
        """Save scaler state."""
        with open(path, 'wb') as f:
            pickle.dump({'scaler': self.scaler, 'log_features': self.log_features, 'feature_columns': self.feature_columns}, f)

    def load(self, path: str):
        """Load scaler state."""
        with open(path, 'rb') as f:
            state = pickle.load(f)
            self.scaler = state['scaler']
            self.log_features = state['log_features']
            self.feature_columns = state['feature_columns']
            self.fitted = True
