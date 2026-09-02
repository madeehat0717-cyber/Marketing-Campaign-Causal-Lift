import pytest
import numpy as np
import pandas as pd
import os
import sys

# Ensure src can be imported
base_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(base_dir)

from src.data_generation import generate_synthetic_data
from src.data_cleaning import clean_data
from src.feature_engineering import engineer_features
from src.causal_analysis import calculate_ate
from src.uplift_model import train_uplift_model
from src.segmentation import segment_customers
from src.roi_simulator import simulate_roi

@pytest.fixture
def sample_data():
    df = generate_synthetic_data(n_samples=1000, random_state=42)
    return df

def test_data_generation(sample_data):
    assert len(sample_data) >= 1000
    assert 'customer_id' in sample_data.columns
    assert 'treatment' in sample_data.columns
    assert 'purchase_outcome' in sample_data.columns

def test_data_cleaning(sample_data):
    df_clean = clean_data(sample_data)
    # Check if duplicates are removed
    assert df_clean['customer_id'].duplicated().sum() == 0
    # Check if NaNs in income are imputed
    assert df_clean['income'].isna().sum() == 0

def test_feature_engineering(sample_data):
    df_clean = clean_data(sample_data)
    df_feat, feature_cols = engineer_features(df_clean)
    assert 'engagement_score' in df_feat.columns
    assert len(feature_cols) > 0

def test_causal_analysis(sample_data):
    results = calculate_ate(sample_data)
    assert 'absolute_lift' in results
    assert 'p_value' in results
    assert results['n_treatment'] > 0
    assert results['n_control'] > 0

def test_uplift_model(sample_data):
    df_clean = clean_data(sample_data)
    df_feat, feature_cols = engineer_features(df_clean)
    model, baseline, df_scored = train_uplift_model(df_feat, feature_cols)
    assert 'pred_ite' in df_scored.columns
    assert 'pred_prob_t1' in df_scored.columns
    assert 'pred_prob_t0' in df_scored.columns

def test_segmentation(sample_data):
    df_clean = clean_data(sample_data)
    df_feat, feature_cols = engineer_features(df_clean)
    _, _, df_scored = train_uplift_model(df_feat, feature_cols)
    df_seg = segment_customers(df_scored)
    assert 'uplift_segment' in df_seg.columns
    segments = df_seg['uplift_segment'].unique()
    assert len(segments) > 0

def test_roi_simulator(sample_data):
    df_clean = clean_data(sample_data)
    df_feat, feature_cols = engineer_features(df_clean)
    _, _, df_scored = train_uplift_model(df_feat, feature_cols)
    roi_results = simulate_roi(df_scored, target_percentage=20, campaign_cost_per_customer=2.0, revenue_per_conversion=100.0)
    assert 'Random' in roi_results
    assert 'Uplift' in roi_results
    assert 'Profit' in roi_results['Uplift']
