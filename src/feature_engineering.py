import pandas as pd
from sklearn.preprocessing import StandardScaler

def engineer_features(df):
    """
    Performs feature engineering on the cleaned dataset.
    """
    df_feat = df.copy()
    
    # Interaction features
    df_feat['engagement_score'] = df_feat['email_open_rate'] * df_feat['website_visits']
    df_feat['value_per_tenure'] = df_feat['average_order_value'] / (df_feat['customer_tenure'] + 1)
    
    # We could do more complex stuff, but keep it light.
    # No target leakage!
    
    # Define feature columns
    feature_cols = [
        'age', 'income', 'customer_tenure', 'previous_purchases',
        'average_order_value', 'website_visits', 'email_open_rate',
        'discount_usage', 'recency_days', 'engagement_score', 'value_per_tenure'
    ]
    
    return df_feat, feature_cols

