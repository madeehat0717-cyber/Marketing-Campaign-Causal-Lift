import os
import sqlite3
import pandas as pd
import sys

base_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(base_dir)

from src.roi_simulator import simulate_roi
from src.evaluation import qini_curve_data, calculate_auuc

def run_verification():
    db_path = os.path.join(base_dir, 'data', 'campaign.db')
    csv_path = os.path.join(base_dir, 'data', 'processed', 'scored_data.csv')
    
    print("--- VERIFICATION REPORT DATA ---")
    
    if not os.path.exists(csv_path):
        print("ERROR: Processed data not found.")
        return
        
    df = pd.read_csv(csv_path)
    
    # 1. ATE and Rates
    t_group = df[df['treatment'] == 1]['purchase_outcome']
    c_group = df[df['treatment'] == 0]['purchase_outcome']
    
    t_rate = t_group.mean()
    c_rate = c_group.mean()
    ate = t_rate - c_rate
    print(f"Treatment Conversion Rate: {t_rate:.4f}")
    print(f"Control Conversion Rate: {c_rate:.4f}")
    print(f"ATE (Absolute Lift): {ate:.4f}")
    
    # 2. Model Used
    print("ML Model Used: HistGradientBoostingClassifier (via T-Learner)")
    
    # 3. Uplift Evaluation Results (AUUC)
    x_axis, qini, random_line = qini_curve_data(df)
    auuc = calculate_auuc(x_axis, qini, random_line)
    print(f"Uplift AUUC (Qini area over random): {auuc:.2f}")
    
    # 4. ROI Comparison
    roi_results = simulate_roi(df, target_percentage=20, campaign_cost_per_customer=2.0, revenue_per_conversion=100.0)
    print("ROI Comparison (Targeting Top 20%):")
    for strategy, metrics in roi_results.items():
        print(f"  {strategy}: ROI = {metrics['ROI (%)']:.2f}%, Profit = ${metrics['Profit']:.2f}")
        
    # 5. SQL Verification
    print("--- SQL VERIFICATION ---")
    conn = sqlite3.connect(db_path)
    sql_file = os.path.join(base_dir, 'sql', 'queries.sql')
    
    with open(sql_file, 'r') as f:
        sql_script = f.read()
        
    # Split queries by semicolon to execute them one by one
    queries = [q.strip() for q in sql_script.split(';') if q.strip()]
    for i, q in enumerate(queries):
        try:
            res = pd.read_sql(q, conn)
            print(f"Query {i+1} OK: returned {len(res)} rows.")
        except Exception as e:
            print(f"ERROR in Query {i+1}: {e}")
            
    conn.close()

if __name__ == '__main__':
    run_verification()
