


import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw dataset: remove duplicates, handle NaNs, outliers."""
    print("Cleaning dataset...")
    
    # Remove duplicates
    df = df.drop_duplicates()
    print(f"After duplicates removal: {df.shape}")
    
    # Handle NaNs (fill with 0 for amounts, mode for categoricals)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    # Remove extreme outliers (99.9th percentile)
    for col in numeric_cols:
        if df[col].std() > 0:
            p999 = df[col].quantile(0.999)
            df = df[df[col] <= p999]
    
    print(f"After outlier removal: {df.shape}")
    return df

def prepare_for_ml(df: pd.DataFrame, target_col: str = "from_scam") -> tuple:
    """Prepare cleaned data for ML training."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Available: {list(df.columns)}")
    
    # Drop non-features (IDs, hashes, categories with too many NaNs)
    drop_cols = ['hash', 'block_hash', 'from_address', 'to_address', 'input', 
                 'from_category', 'to_category']
    available_drop = [col for col in drop_cols if col in df.columns]
    X = df.drop(columns=[target_col] + available_drop)
    y = df[target_col]
    
    print(f"Features shape: {X.shape}")
    print(f"Features: {list(X.columns)}")
    
    # Convert value/gas_price to numeric
    for col in ['value', 'gas_price']:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    
    # One-hot encoding for remaining categoricals (si quedan)
    categorical_cols = X.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        X = pd.get_dummies(X, columns=categorical_cols)
    
    # Scale numeric features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    
    return X, y, scaler