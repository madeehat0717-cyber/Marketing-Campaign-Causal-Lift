import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss

class TLearner:
    """
    Implements a T-learner for uplift modeling.
    Trains two separate models:
    Model 1: P(Y=1 | X, T=1)
    Model 0: P(Y=1 | X, T=0)
    ITE = P(Y=1 | X, T=1) - P(Y=1 | X, T=0)
    """
    def __init__(self, model_class=HistGradientBoostingClassifier, **model_params):
        # We use HistGradientBoostingClassifier for speed and native handling of various distributions
        self.model_t1 = model_class(**model_params)
        self.model_t0 = model_class(**model_params)
        
        # We will calibrate the classifiers to output better probabilities
        self.calibrated_t1 = CalibratedClassifierCV(self.model_t1, method='sigmoid', cv=3)
        self.calibrated_t0 = CalibratedClassifierCV(self.model_t0, method='sigmoid', cv=3)
        self.feature_cols = None
        
    def fit(self, X, y, treatment):
        self.feature_cols = list(X.columns)
        
        X_t1, y_t1 = X[treatment == 1], y[treatment == 1]
        X_t0, y_t0 = X[treatment == 0], y[treatment == 0]
        
        self.calibrated_t1.fit(X_t1, y_t1)
        self.calibrated_t0.fit(X_t0, y_t0)
        
        return self
        
    def predict_uplift(self, X):
        prob_t1 = self.calibrated_t1.predict_proba(X)[:, 1]
        prob_t0 = self.calibrated_t0.predict_proba(X)[:, 1]
        ite = prob_t1 - prob_t0
        return prob_t1, prob_t0, ite

def train_uplift_model(df, feature_cols, target_col='purchase_outcome', treatment_col='treatment'):
    """
    Trains the T-Learner and returns the model and test dataset predictions.
    """
    X = df[feature_cols]
    y = df[target_col]
    treatment = df[treatment_col]
    
    # We want a proper train/test split. Stratify by treatment and outcome.
    # Combine them for stratification
    strat = df[treatment_col].astype(str) + "_" + df[target_col].astype(str)
    
    X_train, X_test, y_train, y_test, t_train, t_test, idx_train, idx_test = train_test_split(
        X, y, treatment, df.index, test_size=0.3, random_state=42, stratify=strat
    )
    
    model = TLearner(model_class=HistGradientBoostingClassifier, random_state=42, max_iter=100)
    model.fit(X_train, y_train, t_train)
    
    # Predict on test set
    prob_t1, prob_t0, ite = model.predict_uplift(X_test)
    
    results_df = df.loc[idx_test].copy()
    results_df['pred_prob_t1'] = prob_t1
    results_df['pred_prob_t0'] = prob_t0
    results_df['pred_ite'] = ite
    
    # Also train a standard purchase probability model (ignoring treatment) as baseline
    baseline_model = HistGradientBoostingClassifier(random_state=42, max_iter=100)
    baseline_calibrated = CalibratedClassifierCV(baseline_model, method='sigmoid', cv=3)
    baseline_calibrated.fit(X_train, y_train)
    
    results_df['pred_baseline_prob'] = baseline_calibrated.predict_proba(X_test)[:, 1]
    
    # Evaluate T-Learner Models on test set
    print("--- Model Validation Metrics ---")
    mask_t1 = (t_test == 1)
    mask_t0 = (t_test == 0)
    
    if mask_t1.sum() > 0:
        auc_t1 = roc_auc_score(y_test[mask_t1], prob_t1[mask_t1])
        brier_t1 = brier_score_loss(y_test[mask_t1], prob_t1[mask_t1])
        print(f"Treatment Model (T=1) - ROC-AUC: {auc_t1:.4f}, Brier Score: {brier_t1:.4f}")
        
    if mask_t0.sum() > 0:
        auc_t0 = roc_auc_score(y_test[mask_t0], prob_t0[mask_t0])
        brier_t0 = brier_score_loss(y_test[mask_t0], prob_t0[mask_t0])
        print(f"Control Model (T=0) - ROC-AUC: {auc_t0:.4f}, Brier Score: {brier_t0:.4f}")
        
    print("Note: Ordinary classification metrics measure predictive accuracy of the outcomes, not the causal Individual Treatment Effect (ITE) directly.")
    
    return model, baseline_calibrated, results_df

from sklearn.inspection import permutation_importance

def extract_feature_importance(model, X_test, y_test, t_test):
    """
    Extracts permutation importance from the trained T-Learner models on the holdout set.
    Since we are modeling treatment and control separately, we calculate the predictive importance 
    for both models and average them to see which features drive the underlying outcome predictions.
    NOTE: This represents predictive importance in the outcome models, which correlates with 
    causal uplift, but is not strictly 'causal feature importance' in isolation.
    """
    mask_t1 = (t_test == 1)
    mask_t0 = (t_test == 0)
    
    # Calculate for Treatment model (using only treated test instances)
    imp_t1 = permutation_importance(
        model.calibrated_t1, X_test[mask_t1], y_test[mask_t1], 
        n_repeats=5, random_state=42, n_jobs=-1
    )
    
    # Calculate for Control model (using only control test instances)
    imp_t0 = permutation_importance(
        model.calibrated_t0, X_test[mask_t0], y_test[mask_t0], 
        n_repeats=5, random_state=42, n_jobs=-1
    )
    
    # Average the mean importance scores
    avg_importance = (imp_t1.importances_mean + imp_t0.importances_mean) / 2.0
    
    fi_df = pd.DataFrame({
        'Feature': X_test.columns,
        'Importance': avg_importance
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
    
    return fi_df
