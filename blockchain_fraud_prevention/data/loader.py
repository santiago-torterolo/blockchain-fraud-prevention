


import pandas as pd
from pathlib import Path


def load_raw_dataset(file_path: str = "data/blte.csv") -> pd.DataFrame:
    """Load raw BLTE fraud dataset."""
    path = Path(file_path)
    df = pd.read_csv(path)
    
    print(f"Raw dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    return df


def explore_dataset(df: pd.DataFrame) -> None:
    """Print dataset info for exploration."""
    print("\nDataset info:")
    print(df.info())
    print("\nMissing values:")
    print(df.isnull().sum())
    
    # Auto-detect fraud column
    fraud_cols = [col for col in df.columns if any(x in col.lower() for x in ['fraud', 'scam', 'label'])]
    if fraud_cols:
        print(f"\nFraud column detected: {fraud_cols[0]}")
        print(df[fraud_cols[0]].value_counts())


if __name__ == "__main__":
    df = load_raw_dataset()
    explore_dataset(df)
