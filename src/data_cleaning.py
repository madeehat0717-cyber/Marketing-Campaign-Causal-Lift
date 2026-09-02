import pandas as pd
import numpy as np

def clean_data(df):
    """
    Cleans the raw dataset: removes duplicates and handles missing values.
    """
    # 1. Remove duplicates
    initial_shape = df.shape
    df_cleaned = df.drop_duplicates(subset=['customer_id'], keep='first').copy()
    
    # 2. Handle missing values
    # Income has missing values. We will impute with median for simplicity.
    if 'income' in df_cleaned.columns:
        median_income = df_cleaned['income'].median()
        df_cleaned['income'] = df_cleaned['income'].fillna(median_income)
        
    return df_cleaned

if __name__ == '__main__':
    import os
    base_dir = os.path.dirname(os.path.dirname(__file__))
    input_path = os.path.join(base_dir, 'data', 'raw', 'campaign_data.csv')
    if os.path.exists(input_path):
        df = pd.read_csv(input_path)
        df_clean = clean_data(df)
        print(f"Cleaned data: {df.shape[0]} -> {df_clean.shape[0]} rows.")
    else:
        print("Raw data not found. Run data_generation.py first.")
