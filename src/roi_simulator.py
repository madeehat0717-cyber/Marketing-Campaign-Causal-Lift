import numpy as np
import pandas as pd

def simulate_roi(df, target_percentage, campaign_cost_per_customer, revenue_per_conversion, 
                 uplift_col='pred_ite', prob_col='pred_baseline_prob', 
                 treatment_col='treatment', outcome_col='purchase_outcome'):
    """
    Simulates the ROI of targeting a specific percentage of the customer base using three strategies:
    1. Random Targeting
    2. Purchase-Probability Targeting (Target highest probability of purchase)
    3. Uplift Targeting (Target highest estimated ITE)
    """
    
    n_total = len(df)
    n_target = int(n_total * (target_percentage / 100.0))
    
    if n_target == 0:
        return {}

    def calculate_metrics(targeted_df):
        # Among the targeted folks, how many were actually treated/control in the historical data?
        t_group = targeted_df[targeted_df[treatment_col] == 1]
        c_group = targeted_df[targeted_df[treatment_col] == 0]
        
        t_rate = t_group[outcome_col].mean() if len(t_group) > 0 else 0
        c_rate = c_group[outcome_col].mean() if len(c_group) > 0 else 0
        
        # Expected conversions if EVERYONE in the targeted group was treated vs not treated
        expected_treated_conversions = n_target * t_rate
        expected_untreated_conversions = n_target * c_rate
        
        incremental_conversions = expected_treated_conversions - expected_untreated_conversions
        incremental_revenue = incremental_conversions * revenue_per_conversion
        campaign_cost = n_target * campaign_cost_per_customer
        profit = incremental_revenue - campaign_cost
        roi = (profit / campaign_cost * 100) if campaign_cost > 0 else 0
        
        return {
            'Targeted Customers': n_target,
            'Expected Conversions (Treated)': expected_treated_conversions,
            'Expected Conversions (Untreated)': expected_untreated_conversions,
            'Incremental Conversions': incremental_conversions,
            'Incremental Revenue': incremental_revenue,
            'Campaign Cost': campaign_cost,
            'Profit': profit,
            'ROI (%)': roi
        }

    # 1. Random Targeting
    np.random.seed(42)
    random_indices = np.random.choice(df.index, size=n_target, replace=False)
    df_random = df.loc[random_indices]
    results_random = calculate_metrics(df_random)
    
    # 2. Purchase-Probability Targeting (Sort by baseline probability descending)
    df_prob = df.sort_values(by=prob_col, ascending=False).head(n_target)
    results_prob = calculate_metrics(df_prob)
    
    # 3. Uplift Targeting (Sort by estimated ITE descending)
    df_uplift = df.sort_values(by=uplift_col, ascending=False).head(n_target)
    results_uplift = calculate_metrics(df_uplift)
    
    return {
        'Random': results_random,
        'Purchase-Probability': results_prob,
        'Uplift': results_uplift
    }

