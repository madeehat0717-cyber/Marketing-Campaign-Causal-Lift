import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def qini_curve_data(df, uplift_col='pred_ite', treatment_col='treatment', outcome_col='purchase_outcome'):
    """
    Calculates the data points for a Qini curve.
    """
    # Sort dataframe by predicted uplift descending
    sorted_df = df.sort_values(by=uplift_col, ascending=False).reset_index(drop=True)
    
    n = len(sorted_df)
    t_group = sorted_df[sorted_df[treatment_col] == 1]
    c_group = sorted_df[sorted_df[treatment_col] == 0]
    
    # Cumulative responses
    t_responses = np.cumsum(sorted_df[treatment_col] * sorted_df[outcome_col])
    c_responses = np.cumsum((1 - sorted_df[treatment_col]) * sorted_df[outcome_col])
    
    # Cumulative counts
    t_counts = np.cumsum(sorted_df[treatment_col])
    c_counts = np.cumsum(1 - sorted_df[treatment_col])
    
    # Avoid division by zero
    t_counts_safe = np.where(t_counts == 0, 1, t_counts)
    c_counts_safe = np.where(c_counts == 0, 1, c_counts)
    
    # Qini formula: U(x) = R_t(x) - R_c(x) * (N_t(x) / N_c(x))
    # where R is cumulative responses and N is cumulative counts
    qini = t_responses - c_responses * (t_counts / c_counts_safe)
    
    # Qini value is 0 at the start (handling division by zero issues at the very beginning)
    qini = np.insert(qini.values, 0, 0)
    
    # Random targeting line (straight line from 0 to total incremental responses)
    total_inc = qini[-1]
    random_line = np.linspace(0, total_inc, n + 1)
    
    x_axis = np.linspace(0, 100, n + 1)
    
    return x_axis, qini, random_line

def calculate_auuc(x_axis, qini, random_line):
    """
    Calculates the Area Under the Uplift Curve (AUUC) relative to random.
    """
    # Area under Qini
    area_qini = np.trapezoid(qini, x_axis)
    # Area under random
    area_random = np.trapezoid(random_line, x_axis)
    
    auuc = area_qini - area_random
    return auuc

def evaluate_deciles(df, uplift_col='pred_ite', treatment_col='treatment', outcome_col='purchase_outcome'):
    """
    Evaluates response rates by deciles of predicted uplift.
    """
    # Sort dataframe by predicted uplift descending
    df_sorted = df.sort_values(by=uplift_col, ascending=False).reset_index(drop=True)
    
    # Create deciles (1 is highest uplift, 10 is lowest)
    df_sorted['decile'] = pd.qcut(df_sorted.index, 10, labels=np.arange(1, 11))
    
    # Group by decile and treatment
    results = []
    for decile in range(1, 11):
        decile_data = df_sorted[df_sorted['decile'] == decile]
        
        t_data = decile_data[decile_data[treatment_col] == 1]
        c_data = decile_data[decile_data[treatment_col] == 0]
        
        t_rate = t_data[outcome_col].mean() if len(t_data) > 0 else 0
        c_rate = c_data[outcome_col].mean() if len(c_data) > 0 else 0
        
        actual_lift = t_rate - c_rate
        avg_pred_uplift = decile_data[uplift_col].mean()
        
        results.append({
            'decile': decile,
            'treatment_rate': t_rate,
            'control_rate': c_rate,
            'actual_lift': actual_lift,
            'predicted_uplift': avg_pred_uplift,
            'size': len(decile_data)
        })
        
    return pd.DataFrame(results)

