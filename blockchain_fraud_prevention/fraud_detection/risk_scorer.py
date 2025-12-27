


import joblib
import numpy as np
import pandas as pd
from typing import Tuple


class RiskScorer:
    def __init__(self, model_path: str = "models/rf_fraud_detector.pkl"):
        """Initialize with trained model."""
        self.full_model = joblib.load(model_path)
        self.model = self.full_model['model']
        self.scaler = self.full_model['scaler']
        self.feature_names = self.full_model['feature_names']
        print(f"Loaded model with features: {self.feature_names}")

    def score_transaction(self, transaction_data: dict) -> Tuple[float, str]:
        """Score single transaction risk (0-100)."""
        # Create DataFrame with ALL expected features
        df = pd.DataFrame()
        for feature in self.feature_names:
            df[feature] = [transaction_data.get(feature, 0.0)]
        
        # Scale features (CRITICAL)
        X_scaled = self.scaler.transform(df[self.feature_names])
        
        # Predict fraud probability
        fraud_prob = self.model.predict_proba(X_scaled)[0, 1]
        
        # Convert to risk score (0-100)
        risk_score = min(fraud_prob * 100, 100)
        
        # Risk level
        if risk_score < 20:
            risk_level = "LOW"
        elif risk_score < 50:
            risk_level = "MEDIUM"
        elif risk_score < 80:
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"
        
        return risk_score, risk_level

    def score_transactions(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """Score multiple transactions."""
        results = []
        for idx, row in transactions_df.iterrows():
            data = row.to_dict()
            score, level = self.score_transaction(data)
            results.append({
                'risk_score': score,
                'risk_level': level,
                'is_fraud_predicted': score > 50
            })
        return pd.DataFrame(results)

    def batch_risk_assessment(self, X: pd.DataFrame) -> pd.DataFrame:
        """Batch scoring for ML features."""
        X_scaled = self.scaler.transform(X[self.feature_names])
        fraud_prob = self.model.predict_proba(X_scaled)[:, 1]
        
        risk_scores = pd.DataFrame({
            'risk_score': fraud_prob * 100,
            'risk_level': pd.cut(fraud_prob * 100, 
                               bins=[0, 20, 50, 80, 100], 
                               labels=['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']),
            'is_fraud_predicted': fraud_prob > 0.5
        })
        return risk_scores


if __name__ == "__main__":
    scorer = RiskScorer()
    high_risk_tx = {
        'velocity': 25.0, 'value_zscore': 2.5, 'gas_efficiency': 0.9,
        'gas_price_zscore': 1.2, 'tx_position_zscore': 0.5, 'nonce_zscore': 0.1,
        'total_scam_risk': 2.0, 'hour': 3, 'night_tx': 1
    }
    score, level = scorer.score_transaction(high_risk_tx)
    print(f"High risk tx: {score:.1f}% ({level})")
