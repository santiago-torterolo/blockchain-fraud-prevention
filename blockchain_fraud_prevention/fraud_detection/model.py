


import joblib
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from typing import Dict
import numpy as np


class FraudDetector:
    def __init__(self):
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.feature_names = None

    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Train Random Forest model."""
        print("Training Random Forest...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.rf_model.fit(X_train, y_train)
        self.feature_names = X.columns.tolist()
        
        # Cross-validation
        cv_scores = cross_val_score(self.rf_model, X_train, y_train, cv=5)
        y_pred = self.rf_model.predict(X_test)
        
        results = {
            'cv_accuracy': cv_scores.mean(),
            'test_accuracy': self.rf_model.score(X_test, y_test),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'feature_importance': dict(zip(X.columns, self.rf_model.feature_importances_))
        }
        
        print(f"CV Accuracy: {results['cv_accuracy']:.3f}")
        print(f"Test Accuracy: {results['test_accuracy']:.3f}")
        
        print("\nTop 5 features:")
        sorted_features = sorted(results['feature_importance'].items(), 
                               key=lambda x: x[1], reverse=True)[:5]
        for feature, importance in sorted_features:
            print(f"  {feature}: {importance:.3f}")
        
        return results

    def save_model(self, path: str = "models/rf_fraud_detector.pkl") -> None:
        """Save trained model."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.rf_model, path)
        print(f"Model saved to {path}")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict fraud probability."""
        if self.feature_names is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Ensure same features
        if list(X.columns) != self.feature_names:
            raise ValueError(f"Expected features {self.feature_names}, got {list(X.columns)}")
        
        return self.rf_model.predict_proba(X)[:, 1]  # Fraud probability
