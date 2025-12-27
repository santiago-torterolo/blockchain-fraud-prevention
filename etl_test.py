from blockchain_fraud_prevention.data.loader import load_raw_dataset
from blockchain_fraud_prevention.data.preprocessor import clean_dataset
from blockchain_fraud_prevention.fraud_detection.feature_engineering import engineer_fraud_features
from blockchain_fraud_prevention.fraud_detection.model import FraudDetector
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd


def run_full_etl():
    print("=== FULL ETL PIPELINE ===\n")
    
    # 1. Load raw
    print("STEP 1: LOADING RAW DATASET")
    df_raw = load_raw_dataset()
    
    # 2. Clean
    print("\nSTEP 2: DATA CLEANING")
    df_clean = clean_dataset(df_raw)
    
    # 3. Engineer fraud features
    print("\nSTEP 3: FRAUD FEATURE ENGINEERING")
    df_features = engineer_fraud_features(df_clean)
    
    # 4. Extract target BEFORE preprocessing
    y = df_clean['from_scam'].iloc[:len(df_features)]
    
    # 5. Prepare features for ML
    X = df_features.fillna(0)
    
    # Scale features
    print("\nSTEP 4: FEATURE SCALING")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)
    
    # 6. ML Training
    print("\nSTEP 5: ML TRAINING")
    detector = FraudDetector()
    results = detector.train(X, y)
    detector.save_model()
    
    print("\n=== ETL + ML COMPLETE ===")
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Fraud rate: {y.mean():.2%}")
    print(f"Model CV Accuracy: {results['cv_accuracy']:.3f}")
    print(f"Model Test Accuracy: {results['test_accuracy']:.3f}")
    
    return X, y, scaler, results


if __name__ == "__main__":
    X, y, scaler, results = run_full_etl()
