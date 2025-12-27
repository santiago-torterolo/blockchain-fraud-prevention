"""
Blockchain Fraud Prevention System

A machine learning-powered fraud detection system for blockchain transactions.
99.1% accuracy on 70,828 Ethereum transactions using Random Forest classification.

Components:
- blockchain: Core blockchain implementation (POW, validation)
- fraud_detection: ML models and risk scoring
- api: Streamlit dashboard for visualization and interaction
- data: Transaction datasets and preprocessing
"""

__version__ = "1.0.0"
__author__ = "Santiago Torterolo"

from blockchain_fraud_prevention.fraud_detection.model import FraudDetector
from blockchain_fraud_prevention.fraud_detection.risk_scorer import RiskScorer

__all__ = ["FraudDetector", "RiskScorer"]
