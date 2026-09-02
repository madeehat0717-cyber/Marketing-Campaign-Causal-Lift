import pandas as pd
import numpy as np
from scipy import stats

def calculate_ate(df, treatment_col='treatment', outcome_col='purchase_outcome'):
    """
    Calculates the Average Treatment Effect (ATE) and performs statistical testing.
    Assumes randomized treatment assignment (strong ignorability).
    """
    treatment_group = df[df[treatment_col] == 1][outcome_col]
    control_group = df[df[treatment_col] == 0][outcome_col]
    
    treatment_rate = treatment_group.mean()
    control_rate = control_group.mean()
    
    n_treatment = len(treatment_group)
    n_control = len(control_group)
    
    absolute_lift = treatment_rate - control_rate
    relative_lift = absolute_lift / control_rate if control_rate > 0 else 0
    
    # Statistical significance (two-proportion z-test)
    # Using scipy's t-test as an approximation or proportions_ztest from statsmodels
    # We will use scipy.stats.ttest_ind which is acceptable for large samples of binary data
    t_stat, p_value = stats.ttest_ind(treatment_group, control_group, equal_var=False)
    
    # Standard error of the difference
    se_diff = np.sqrt( (treatment_rate * (1 - treatment_rate) / n_treatment) + 
                       (control_rate * (1 - control_rate) / n_control) )
    
    # 95% Confidence Interval for Absolute Lift
    ci_lower = absolute_lift - 1.96 * se_diff
    ci_upper = absolute_lift + 1.96 * se_diff
    
    # Positivity/Overlap Diagnostic
    # Verify that treatment assignment is truly random and bounded away from 0 and 1
    # We do a quick propensity score check using basic features
    from sklearn.linear_model import LogisticRegression
    # Use simple numeric features for quick diagnostic
    diagnostic_features = df.select_dtypes(include=[np.number]).drop(columns=[treatment_col, outcome_col, 'revenue', 'true_prob_y0', 'true_prob_y1', 'true_ite'], errors='ignore').fillna(0)
    
    ps_model = LogisticRegression(max_iter=1000)
    ps_model.fit(diagnostic_features, df[treatment_col])
    propensity_scores = ps_model.predict_proba(diagnostic_features)[:, 1]
    ps_min, ps_max = propensity_scores.min(), propensity_scores.max()
    
    results = {
        'treatment_conversion_rate': treatment_rate,
        'control_conversion_rate': control_rate,
        'absolute_lift': absolute_lift,
        'relative_lift': relative_lift,
        'p_value': p_value,
        'significant_at_5_pct': p_value < 0.05,
        'ci_lower_95': ci_lower,
        'ci_upper_95': ci_upper,
        'n_treatment': n_treatment,
        'n_control': n_control,
        'propensity_score_min': ps_min,
        'propensity_score_max': ps_max
    }
    
    return results

