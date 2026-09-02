import numpy as np
import pandas as pd

def segment_customers(df, ite_col='pred_ite', prob_t0_col='pred_prob_t0'):
    """
    Segments customers based on uplift predictions.
    Segments:
    - Persuadables: High ITE
    - Sure Things: High prob_t0, Low ITE
    - Lost Causes: Low prob_t0, Low ITE
    - Do-Not-Disturb (Sleeping Dogs): Negative ITE
    """
    df_seg = df.copy()
    
    # We will define thresholds based on percentiles to avoid hardcoding absolute values.
    # But for negative ITE, it's explicitly negative.
    
    ite = df_seg[ite_col]
    prob_t0 = df_seg[prob_t0_col]
    
    # Thresholds
    # Persuadables: ITE > 75th percentile of positive ITEs or simply high ITE
    ite_75 = np.percentile(ite, 75)
    
    # Sure Things: High base probability (e.g. top 25%), relatively low ITE
    prob_t0_75 = np.percentile(prob_t0, 75)
    
    conditions = [
        (ite < 0), # Do Not Disturb
        (ite >= ite_75), # Persuadables (High uplift)
        ((prob_t0 >= prob_t0_75) & (ite >= 0) & (ite < ite_75)), # Sure things
        ((prob_t0 < prob_t0_75) & (ite >= 0) & (ite < ite_75)) # Lost causes
    ]
    
    choices = ['Do-Not-Disturb', 'Persuadables', 'Sure Things', 'Lost Causes']
    
    df_seg['uplift_segment'] = np.select(conditions, choices, default='Lost Causes')
    
    return df_seg

