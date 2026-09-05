# Project Documentation: Causal Inference & Uplift Modeling

This document provides explanations of the core methodological concepts used in this educational portfolio project.

## Correlation vs Causation
- **What it is:** Correlation means two things happen together. Causation means one thing *makes* the other happen.
- **Why we used it:** We need to know if the marketing campaign *caused* the purchase, not just if campaign recipients bought more.
- **In this project:** We estimate the causal effect of the campaign (treatment) on the purchase outcome using simulated data.

## A/B Testing, Treatment, and Control
- **What it is:** Splitting a population into a group that receives a campaign (Treatment) and a group that does not (Control).
- **Why we used it:** Randomized assignment supports the ignorability assumption (uncounfoundedness), ensuring both groups are comparable.
- **In this project:** The synthetic data generates a strictly randomized treatment assignment (propensity score ~ 0.5) to ensure causal estimates are valid. Positivity/overlap is verified analytically.

## Average Treatment Effect (ATE)
- **What it is:** The average difference in outcomes between the treatment and control groups across the entire population.
- **Why we used it:** To evaluate the overall historical campaign impact before attempting individual-level modeling.
- **In this project:** It's the experimental overall lift: E[Y|T=1] - E[Y|T=0].

## Individual Treatment Effect (ITE) / Conditional Average Treatment Effect (CATE)
- **What it is:** The expected causal effect of the treatment on a *specific individual* conditional on their covariates X. 
- **Why we used it:** To target customers who have heterogeneous treatment effects (e.g., some react well, others ignore or get annoyed).
- **In this project:** We estimate model-based ITE using a T-Learner. True ITE exists in the simulated ground-truth but is *never* used as a model feature.

## Uplift Modeling (T-Learner)
- **What it is:** A machine learning framework for estimating ITE. A T-Learner (Two-Learner) trains one model on the treatment group and one on the control group.
- **How it works:** Estimated ITE = P(Y=1|X, T=1) - P(Y=1|X, T=0).
- **Assumptions:** Causal interpretation requires strong ignorability (no unmeasured confounders) and SUTVA (Stable Unit Treatment Value Assumption, meaning no interference between users). These are methodological assumptions that we satisfy by design in this synthetic simulation.

## Customer Segmentation
Customers are scored and segmented based on model predictions:
- **Persuadables:** High ITE. They buy *because* of the campaign. Target them!
- **Sure Things:** High baseline probability P(Y=1|T=0, X) but low/zero ITE. Don't waste money targeting them.
- **Lost Causes:** Low baseline probability and low/zero ITE. Unaffected by the campaign.
- **Do-Not-Disturb:** Negative ITE (P(Y=1|T=1, X) < P(Y=1|T=0, X)). The campaign makes them *less* likely to buy.

## Evaluation Metrics (Qini Curve and AUUC)
- **What it is:** The Qini curve plots the cumulative incremental purchases: $R_t(x) - R_c(x) \times \frac{N_t(x)}{N_c(x)}$ against the targeted population fraction. The Qini Coefficient is the Area Under the Qini Curve (AUUC) minus the area under a random targeting baseline.
- **Why we used it:** Ordinary classification metrics (like ROC-AUC) measure predictive accuracy of the *outcome*, not the causal *incremental impact*.
- **In this project:** We report the Qini Coefficient to demonstrate how the uplift model outperforms a random targeting baseline under the simulated campaign assumptions.

## Explainable AI (Feature Importance)
- **What it is:** We extract Permutation Importance from the calibrated outcome models.
- **Why we used it:** To understand which customer attributes most strongly influence the underlying purchase probability models that determine the ITE.
- **In this project:** It is strictly documented that this represents *predictive importance of the outcome*, which correlates with, but does not definitively establish, causal feature importance.

## Uplift Decile Lift Analysis
- **What it is:** Grouping customers by their predicted uplift into 10 equal buckets, and then observing the *actual empirical difference* between the treatment and control outcomes in those buckets.
- **Why we used it:** To visually prove that the customers the model ranked highest actually exhibited the highest true incremental conversion rates.

## Target Export
- **What it is:** The Customer Targeting tab allows marketing managers to dynamically filter segments (e.g. "Persuadables") and download a CSV list for production use.
