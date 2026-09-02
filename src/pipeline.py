import os
import pandas as pd
from data_generation import generate_synthetic_data
from data_cleaning import clean_data
from feature_engineering import engineer_features
from causal_analysis import calculate_ate
from uplift_model import train_uplift_model
from segmentation import segment_customers
import joblib

def run_pipeline():
    print("Starting Data Science Pipeline...")
    base_dir = os.path.dirname(os.path.dirname(__file__))
    raw_dir = os.path.join(base_dir, 'data', 'raw')
    processed_dir = os.path.join(base_dir, 'data', 'processed')
    models_dir = os.path.join(base_dir, 'models')
    
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Generate Data
    print("1. Generating synthetic data...")
    df_raw = generate_synthetic_data(n_samples=20000, output_dir=raw_dir)
    
    # 2. Clean Data
    print("2. Cleaning data...")
    df_clean = clean_data(df_raw)
    
    # 3. Feature Engineering
    print("3. Engineering features...")
    df_feat, feature_cols = engineer_features(df_clean)
    
    # 4. Causal Analysis (ATE)
    print("4. Calculating overall campaign effect (ATE)...")
    ate_results = calculate_ate(df_feat)
    print(f"ATE Results: Absolute Lift: {ate_results['absolute_lift']:.4f}, p-value: {ate_results['p_value']:.4f}")
    print(f"Positivity Diagnostic: Propensity Scores range from {ate_results['propensity_score_min']:.4f} to {ate_results['propensity_score_max']:.4f}")
    
    # 5. Train Uplift Model
    print("5. Training Uplift Model (T-Learner)...")
    uplift_model, baseline_model, df_scored = train_uplift_model(df_feat, feature_cols)
    
    # Save models
    joblib.dump(uplift_model, os.path.join(models_dir, 'uplift_model.joblib'))
    joblib.dump(baseline_model, os.path.join(models_dir, 'baseline_model.joblib'))
    joblib.dump(feature_cols, os.path.join(models_dir, 'feature_cols.joblib'))
    
    # 6. Customer Segmentation
    print("6. Segmenting customers...")
    df_final = segment_customers(df_scored)
    
    # Save processed and scored data
    final_path = os.path.join(processed_dir, 'scored_data.csv')
    df_final.to_csv(final_path, index=False)
    print(f"Pipeline complete. Scored data saved to {final_path}")
    
    # 7. Setup SQLite
    print("7. Setting up SQLite database...")
    import sys
    sys.path.append(base_dir)
    from sql.setup_db import load_data_to_sqlite
    db_path = os.path.join(base_dir, 'data', 'campaign.db')
    load_data_to_sqlite(final_path, db_path)
    
    print("All done!")

if __name__ == '__main__':
    run_pipeline()
