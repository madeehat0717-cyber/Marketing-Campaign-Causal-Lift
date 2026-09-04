# Marketing Campaign Causal Lift & Uplift Modeling Platform

## 1. Project Title
**Marketing Campaign Causal Lift & Uplift Modeling Platform**

🚀 **[View the Live Interactive Dashboard Here!](https://marketing-campaign-causal-lift.streamlit.app/)** *(Note: Replace this link with your actual Streamlit Cloud URL once deployed)*

## 2. Problem Statement
Ordinary machine learning models in marketing predict *\"Who is likely to purchase?\"* 
This project answers a more valuable business question: *\"Who is likely to purchase BECAUSE they received the marketing campaign?\"*

## 3. Why Ordinary Prediction is Insufficient
If we only target customers with a high probability of purchasing, we end up spending marketing budget on **\"Sure Things\"** — people who would have bought anyway. We might also annoy people who are highly engaged but react negatively to spam (**\"Do-Not-Disturb\"**). 

## 4. What Causal Lift Means
Causal lift (or uplift) is the incremental impact of a treatment (like an email campaign) on an individual's behavior. We estimate the Conditional Average Treatment Effect (CATE) or Individual Treatment Effect (ITE):
`ITE = P(Purchase | X, Treatment=1) - P(Purchase | X, Treatment=0)`

## 5. Architecture
This project follows a clean, modular architecture separating the data pipeline, causal analysis, uplift modeling, business logic, and UI. See [ARCHITECTURE.md](ARCHITECTURE.md) for details.

## 6. Dataset (Synthetic)
**Disclaimer:** This is an educational/portfolio project. 
The dataset is **synthetic** and generated to simulate realistic e-commerce behavior, including heterogeneous treatment effects, noise, and overlap. Randomized treatment is simulated. The ground-truth treatment effect is used *only* for validation and is never passed to the model.

## 7. Features
- **Data Generation:** Creates realistic treatment/control data.
- **Causal Analysis:** Evaluates overall experimental ATE using A/B testing principles and verifies overlap/positivity.
- **Uplift Modeling:** Implements a T-Learner using Gradient Boosting, validated strictly on a holdout test set.
- **Customer Segmentation:** Divides customers into Persuadables, Sure Things, Lost Causes, and Do-Not-Disturb.
- **Evaluation:** Qini curves, AUUC (Qini Coefficient), and decile lift analysis.
- **ROI Simulator:** Simulates targeting profitability.
- **Streamlit Dashboard:** Interactive UI.

## 8. Methodology
- **T-Learner:** Two separate machine learning models are trained (one on the treatment group, one on the control group) to estimate counterfactual outcomes.
- **Calibration:** Models are calibrated using `CalibratedClassifierCV` to ensure probabilities are realistic.
- **Validation:** Model metrics are reported for a 30% hold-out test set to prevent train/test contamination.

## 9. EDA & Causal Analysis
See the Jupyter Notebook `notebooks/01_eda_and_causal_analysis.ipynb` (to be created by the user or run locally) and the dashboard for Exploratory Data Analysis.

## 10. Uplift Modeling & 11. Customer Segmentation
Customers are scored and placed into 4 segments based on their base purchase probability and estimated uplift.

## 12. Evaluation
We evaluate the model using the Qini curve (which measures incremental gains over random targeting). 

## 13. ROI Simulator
We simulate targeting the top 20% of customers using:
1. Random Targeting
2. Purchase-Probability Targeting
3. Uplift Targeting

The simulator *demonstrates under the simulated campaign assumptions* that uplift targeting yields the highest ROI. ROI results are simulations, not real company results.

## 14. Installation
```bash
git clone <repo-url>
cd marketing-causal-lift
pip install -r requirements.txt
```

## 15. Running Instructions
```bash
# 1. Run the data pipeline and train the models
python src/pipeline.py

# 2. Run tests
pytest tests/

# 3. Start the dashboard
streamlit run app/streamlit_app.py
```

## 16. Project Structure
- `src/`: Core Python pipeline (data generation, cleaning, features, models).
- `app/`: Streamlit application.
- `sql/`: SQLite database setup and queries.
- `tests/`: Pytest test suite.
- `data/`: Raw and processed data (generated at runtime).
- `models/`: Serialized models.

## 17. Limitations & Assumptions
- Causal estimates depend strictly on experimental design and assumptions.
- **SUTVA and Ignorability** are methodological assumptions that we enforce by design in this synthetic simulation, but which must be rigorously tested in real-world scenarios.
- Individual treatment effects are *model estimates*, not directly observed individual truths.
