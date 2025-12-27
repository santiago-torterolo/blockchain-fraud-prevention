


import pandas as pd
import numpy as np
from typing import Tuple


def engineer_fraud_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer fraud detection features from raw Ethereum transactions."""
    print("Engineering fraud features...")
    
    df = df.copy()
    
    # 1. Convertir timestamps (ISO8601 con timezone)
    df['block_timestamp'] = pd.to_datetime(df['block_timestamp'], format='mixed')
    df['hour'] = df['block_timestamp'].dt.hour
    
    # 2. Velocity (txs por sender en ventana temporal)
    df['timestamp_diff'] = df.groupby('from_address')['block_timestamp'].diff().dt.total_seconds()
    df['velocity'] = 3600 / df['timestamp_diff'].clip(lower=1.0).fillna(3600)
    
    # 3. Amount features
    df['value_numeric'] = pd.to_numeric(df['value'], errors='coerce').fillna(0)
    
    # Z-score por sender (safe division)
    sender_stats = df.groupby('from_address')['value_numeric'].agg(['mean', 'std'])
    sender_stats['std'] = sender_stats['std'].fillna(1.0)  # Avoid division by zero
    df['sender_mean_value'] = df['from_address'].map(sender_stats['mean'])
    df['sender_std_value'] = df['from_address'].map(sender_stats['std'])
    df['value_zscore'] = (df['value_numeric'] - df['sender_mean_value'].fillna(0)) / df['sender_std_value']
    
    # 4. Gas features
    df['gas_efficiency'] = df['receipt_gas_used'] / df['gas'].clip(lower=1.0)
    df['gas_price_gwei'] = pd.to_numeric(df['gas_price'], errors='coerce').fillna(0) / 1e9
    
    gas_mean = df['gas_price_gwei'].mean()
    gas_std = max(df['gas_price_gwei'].std(), 1.0)
    df['gas_price_zscore'] = (df['gas_price_gwei'] - gas_mean) / gas_std
    
    # 5. Transaction features
    df['tx_position_zscore'] = (df['transaction_index'] - df['transaction_index'].mean()) / df['transaction_index'].std()
    df['nonce_zscore'] = (df['nonce'] - df['nonce'].mean()) / df['nonce'].std()
    
    # 6. Scam risk + temporal
    df['total_scam_risk'] = df['from_scam'] + df['to_scam']
    df['night_tx'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
    
    # Select engineered features (ML ready)
    feature_cols = [
        'velocity', 'value_zscore', 'gas_efficiency', 'gas_price_zscore',
        'tx_position_zscore', 'nonce_zscore', 'total_scam_risk', 
        'hour', 'night_tx'
    ]
    
    features_df = df[feature_cols].fillna(0)
    print(f"Engineered features: {features_df.shape}")
    print(f"Feature stats:\n{features_df.describe().round(2)}")
    
    return features_df
