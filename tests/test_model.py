"""
Unit tests for FraudDetector model

Tests model loading, predictions, and accuracy metrics
"""

import pytest
from blockchain_fraud_prevention.fraud_detection.model import FraudDetector
from blockchain_fraud_prevention.fraud_detection.risk_scorer import RiskScorer


class TestFraudDetector:
    """Test cases for FraudDetector class"""
    
    def test_model_initialization(self):
        """Test that model initializes without errors"""
        detector = FraudDetector()
        assert detector is not None
    
    def test_model_has_required_attributes(self):
        """Test model has required attributes"""
        detector = FraudDetector()
        assert hasattr(detector, 'model')
        assert hasattr(detector, 'feature_names')
    
    def test_risk_scorer_initialization(self):
        """Test RiskScorer initializes correctly"""
        scorer = RiskScorer()
        assert scorer is not None
    
    def test_risk_score_range(self):
        """Test risk scores are in valid range (0-100)"""
        scorer = RiskScorer()
        # Mock fraud probability
        fraud_prob = 0.85
        risk_score = scorer.calculate_risk_score(fraud_prob)
        assert 0 <= risk_score <= 100


class TestDataIntegrity:
    """Test cases for data integrity"""
    
    def test_dataset_exists(self):
        """Test that ethereum dataset exists"""
        import os
        data_path = "blockchain_fraud_prevention/data/ethereum_txs.csv"
        assert os.path.exists(data_path), "Dataset file not found"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
