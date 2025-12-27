from blockchain_fraud_prevention.fraud_detection.risk_scorer import RiskScorer
import pandas as pd
import numpy as np

def test_risk_scorer():
    print("Testing RiskScorer...\n")
    
    scorer = RiskScorer()
    
    # Test 1: High-risk (SCALED values ~ +2/+3 std devs)
    print("TEST 1: Single high-risk transaction")
    high_risk_tx = {
        'velocity': 2.5,           # +2.5 std (high velocity)
        'value_zscore': 3.0,       # +3 std (huge amount)
        'gas_efficiency': 1.0,     # Perfect efficiency
        'gas_price_zscore': 2.0,   # +2 std gas price
        'tx_position_zscore': 1.5,
        'nonce_zscore': 1.0,
        'total_scam_risk': 2.0,    # Max scam risk
        'hour': 0.0,               # Midnight
        'night_tx': 1.0
    }
    
    score1, level1 = scorer.score_transaction(high_risk_tx)
    print(f"  High risk: {score1:.1f}% ({level1})")
    
    # Test 2: Low-risk (SCALED values ~ 0)
    print("\nTEST 2: Single low-risk transaction")
    low_risk_tx = {
        'velocity': 0.0,
        'value_zscore': 0.0,
        'gas_efficiency': 0.0,
        'gas_price_zscore': 0.0,
        'tx_position_zscore': 0.0,
        'nonce_zscore': 0.0,
        'total_scam_risk': 0.0,
        'hour': 0.0,
        'night_tx': 0.0
    }
    
    score2, level2 = scorer.score_transaction(low_risk_tx)
    print(f"  Low risk: {score2:.1f}% ({level2})")
    
    # Test 3: Extreme cases
    print("\nTEST 3: Extreme cases")
    extreme_cases = pd.DataFrame([
        # CRITICAL: All max values
        {'velocity': 3.0, 'value_zscore': 3.0, 'gas_efficiency': 1.0, 'gas_price_zscore': 3.0, 
         'tx_position_zscore': 2.0, 'nonce_zscore': 2.0, 'total_scam_risk': 2.0, 'hour': -1.5, 'night_tx': 1.0},
        # HIGH: Moderate anomalies
        {'velocity': 1.5, 'value_zscore': 1.5, 'gas_efficiency': 0.8, 'gas_price_zscore': 1.5, 
         'tx_position_zscore': 1.0, 'nonce_zscore': 0.5, 'total_scam_risk': 1.0, 'hour': -1.0, 'night_tx': 1.0},
    ])
    
    batch_scores = scorer.score_transactions(extreme_cases)
    print(batch_scores[['risk_score', 'risk_level']].round(1))
    
    print("\n✅ All tests passed!")

if __name__ == "__main__":
    test_risk_scorer()
