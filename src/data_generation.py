import numpy as np
import pandas as pd
import os

def generate_synthetic_data(n_samples=20000, random_state=42, output_dir='data/raw'):
    """
    Generates synthetic e-commerce data with heterogeneous treatment effects.
    """
    np.random.seed(random_state)
    
    # 1. Base Customer Features (Covariates X)
    customer_id = [f'C{i:06d}' for i in range(1, n_samples + 1)]
    age = np.random.randint(18, 70, n_samples)
    income = np.random.lognormal(mean=np.log(60000), sigma=0.5, size=n_samples)
    income = np.clip(income, 20000, 200000)
    customer_tenure = np.random.randint(1, 60, n_samples) # in months
    previous_purchases = np.random.poisson(lam=customer_tenure / 10 + 1)
    average_order_value = np.random.normal(loc=100, scale=30, size=n_samples)
    average_order_value = np.clip(average_order_value, 10, 500)
    website_visits = np.random.poisson(lam=previous_purchases * 2 + 1)
    email_open_rate = np.random.beta(a=2, b=5, size=n_samples)
    discount_usage = np.random.beta(a=1, b=3, size=n_samples) # rate of using discounts previously
    recency_days = np.random.randint(1, 365, n_samples)
    
    # 2. Treatment Assignment
    # Randomized treatment assignment (50% treatment, 50% control)
    # Satisfies the strong ignorability assumption (uncounfoundedness) since it's randomly assigned.
    treatment = np.random.binomial(n=1, p=0.5, size=n_samples)
    
    # 3. Generating the Potential Outcomes (Y(0) and Y(1))
    # Base probability of purchase without treatment Y(0)
    base_logits = (
        -3.0 
        + 0.05 * previous_purchases
        - 0.005 * recency_days
        + 0.5 * email_open_rate
        + np.log(income / 50000) * 0.2
    )
    prob_y0 = 1 / (1 + np.exp(-base_logits))
    
    # Treatment Effect (Heterogeneous Conditional Treatment Effect - CATE)
    # The campaign is a discount/promo email.
    treatment_effect = (
        0.05 
        + 0.15 * discount_usage 
        + 0.10 * email_open_rate
        - 0.25 * (prob_y0 > 0.6).astype(int) # Cannibalization/annoyance for sure things
        - 0.15 * (age > 60).astype(int) # Older segment responds much less, perhaps annoyed by digital campaign
    )
    
    # Add some noise to treatment effect
    treatment_effect += np.random.normal(0, 0.02, n_samples)
    
    # Probability of purchase with treatment Y(1)
    prob_y1 = prob_y0 + treatment_effect
    prob_y1 = np.clip(prob_y1, 0, 1)
    
    # 4. Observed Outcome Y
    # SUTVA assumption: no interference between units
    prob_actual = prob_y1 * treatment + prob_y0 * (1 - treatment)
    purchase_outcome = np.random.binomial(n=1, p=prob_actual)
    
    # Revenue (simplified: if purchased, revenue is around AOV)
    revenue = purchase_outcome * np.random.normal(average_order_value, 10)
    revenue = np.clip(revenue, 0, None)
    
    # 5. Missing Values & Noise
    income_missing_idx = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    income_with_nan = income.copy()
    income_with_nan[income_missing_idx] = np.nan
    
    # 6. Construct DataFrame
    df = pd.DataFrame({
        'customer_id': customer_id,
        'age': age,
        'income': income_with_nan,
        'customer_tenure': customer_tenure,
        'previous_purchases': previous_purchases,
        'average_order_value': average_order_value,
        'website_visits': website_visits,
        'email_open_rate': email_open_rate,
        'discount_usage': discount_usage,
        'recency_days': recency_days,
        'treatment': treatment,
        'purchase_outcome': purchase_outcome,
        'revenue': revenue,
        # Ground truth values for testing/validation (should NOT be used as features!)
        'true_prob_y0': prob_y0,
        'true_prob_y1': prob_y1,
        'true_ite': prob_y1 - prob_y0
    })
    
    # Duplicate some rows to simulate dirty data
    duplicates = df.sample(n=int(n_samples * 0.02), random_state=random_state)
    df = pd.concat([df, duplicates], ignore_index=True)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'campaign_data.csv')
    df.to_csv(filepath, index=False)
    print(f"Generated {len(df)} records and saved to {filepath}")
    return df

if __name__ == '__main__':
    # When run directly from src directory
    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw')
    generate_synthetic_data(output_dir=output_path)
