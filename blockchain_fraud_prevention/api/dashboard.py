


import streamlit as st
import pandas as pd
import sys
import os
import joblib

# Fix path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from blockchain_fraud_prevention.data.loader import load_raw_dataset
from blockchain_fraud_prevention.data.preprocessor import clean_dataset
from blockchain_fraud_prevention.fraud_detection.feature_engineering import engineer_fraud_features
from blockchain_fraud_prevention.fraud_detection.risk_scorer import RiskScorer

st.set_page_config(page_title="Blockchain Fraud Detection", layout="wide")

st.title("Blockchain Fraud Detection Dashboard")
st.markdown("Real-time risk scoring for Ethereum transactions")

# Sidebar: Demo dataset
st.sidebar.header("Quick Demo")
if st.sidebar.button("Load Demo Dataset (71k txs)"):
    st.session_state.demo_data = True

# Load model
@st.cache_resource
def load_model():
    return RiskScorer()

scorer = load_model()

# Load/process data
if 'df_features' not in st.session_state:
    st.session_state.df_features = None

# File upload OR demo dataset
uploaded_file = st.file_uploader("Upload transactions CSV", type="csv")

if uploaded_file or st.session_state.get('demo_data'):
    if uploaded_file:
        df_raw = pd.read_csv(uploaded_file)
    else:
        df_raw = load_raw_dataset()
        st.info("Loaded demo dataset: 71,250 Ethereum transactions")
    
    with st.spinner("Processing data (ETL + Feature Engineering)..."):
        df_clean = clean_dataset(df_raw)
        df_features = engineer_fraud_features(df_clean)
    
    st.session_state.df_features = df_features
    st.success(f"✅ Processed {len(df_features)} transactions")
    
    if st.button("Analyze Fraud Risk"):
        risk_scores = scorer.batch_risk_assessment(df_features)
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Transactions", len(df_features))
        col2.metric("High Risk", len(risk_scores[risk_scores['risk_level'] == 'HIGH']))
        col3.metric("Critical Risk", len(risk_scores[risk_scores['risk_level'] == 'CRITICAL']))
        col4.metric("Predicted Fraud Rate", f"{risk_scores['is_fraud_predicted'].mean():.1%}")
        
        # Risk distribution
        st.subheader("Risk Distribution")
        risk_dist = risk_scores['risk_level'].value_counts()
        st.bar_chart(risk_dist)
        
        # Top risky
        st.subheader("Top 10 Riskiest Transactions")
        risky_txs = risk_scores.sort_values('risk_score', ascending=False).head(10)
        st.dataframe(risky_txs)
