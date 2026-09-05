import streamlit as st
import pandas as pd
import numpy as np
import os
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import sys

# Ensure src can be imported
base_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(base_dir)

from src.evaluation import qini_curve_data, evaluate_deciles
from src.roi_simulator import simulate_roi

st.set_page_config(page_title="Marketing Campaign Causal Lift Platform", layout="wide")

# --- Helper Functions ---
@st.cache_data
def load_data():
    csv_path = os.path.join(base_dir, 'data', 'processed', 'scored_data.csv')
    if not os.path.exists(csv_path):
        return None
    return pd.read_csv(csv_path)

@st.cache_data
def load_db():
    db_path = os.path.join(base_dir, 'data', 'campaign.db')
    if not os.path.exists(db_path):
        return None
    return sqlite3.connect(db_path)

@st.cache_data
def load_feature_importance():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    path = os.path.join(base_dir, 'models', 'feature_importance.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

df = load_data()
fi_df = load_feature_importance()

if df is None:
    st.error("Processed data not found. Please run `python src/pipeline.py` first.")
    st.stop()

# --- Title ---
st.title("Marketing Campaign Causal Lift & Uplift Modeling Platform")
st.markdown("""
This platform estimates the **Individual Treatment Effect (ITE)** of a marketing campaign to target 
customers who are most likely to purchase *because* of the campaign, avoiding those who would purchase anyway.
""")

# --- Sidebar / Tabs ---
tabs = st.tabs([
    "Dashboard", 
    "Campaign Analysis", 
    "Uplift Analysis", 
    "Customer Targeting", 
    "ROI Simulator",
    "Customer Explorer"
])

# --- Tab 1: Dashboard ---
with tabs[0]:
    st.header("Campaign KPI Dashboard")
    
    t_group = df[df['treatment'] == 1]
    c_group = df[df['treatment'] == 0]
    
    t_rate = t_group['purchase_outcome'].mean()
    c_rate = c_group['purchase_outcome'].mean()
    absolute_lift = t_rate - c_rate
    
    n_treated = len(t_group)
    n_control = len(c_group)
    
    incremental_conversions = n_treated * absolute_lift
    # Revenue approx
    avg_rev_per_conv = df[df['purchase_outcome'] == 1]['revenue'].mean()
    incremental_revenue = incremental_conversions * avg_rev_per_conv
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{len(df):,}")
    col2.metric("Treatment Customers", f"{n_treated:,}")
    col3.metric("Control Customers", f"{n_control:,}")
    col4.metric("Avg Order Value", f"${df['average_order_value'].mean():.2f}")
    
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Treatment Conv. Rate", f"{t_rate*100:.2f}%")
    col2.metric("Control Conv. Rate", f"{c_rate*100:.2f}%")
    col3.metric("Overall Lift (ATE)", f"{absolute_lift*100:.2f}%")
    col4.metric("Est. Incremental Revenue", f"${incremental_revenue:,.2f}")
    
# --- Tab 2: Campaign Analysis ---
with tabs[1]:
    st.header("Experimental Causal Analysis")
    st.markdown("""
    Before individual-level modeling, we analyze the Average Treatment Effect (ATE). 
    Since treatment was randomly assigned, we can calculate causal impact via A/B testing methods.
    """)
    
    # Statistical Test
    t_stat, p_value = stats.ttest_ind(t_group['purchase_outcome'], c_group['purchase_outcome'], equal_var=False)
    
    se_diff = np.sqrt( (t_rate * (1 - t_rate) / n_treated) + (c_rate * (1 - c_rate) / n_control) )
    ci_lower = absolute_lift - 1.96 * se_diff
    ci_upper = absolute_lift + 1.96 * se_diff
    
    st.info(f"**Statistical Significance:** p-value = {p_value:.4f}. "
            f"Result is {'significant' if p_value < 0.05 else 'not significant'} at 5% level.")
    st.info(f"**95% Confidence Interval for Lift:** [{ci_lower*100:.2f}%, {ci_upper*100:.2f}%]")
    
    fig = px.bar(
        x=['Control', 'Treatment'], 
        y=[c_rate, t_rate], 
        color=['Control', 'Treatment'],
        labels={'x': 'Group', 'y': 'Conversion Rate'},
        title='Treatment vs Control Conversion Rate'
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Tab 3: Uplift Analysis ---
with tabs[2]:
    st.header("Uplift Modeling & Segmentation")
    st.markdown("""
    Using a **T-Learner** (Two-Model approach), we predict:
    1. Probability of purchase *with* treatment
    2. Probability of purchase *without* treatment
    3. Individual Treatment Effect (ITE) = $P(Y=1 | T=1) - P(Y=1 | T=0)$
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Uplift Distribution")
        fig = px.histogram(df, x='pred_ite', nbins=50, title='Estimated ITE Distribution')
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Customer Segments")
        segment_counts = df['uplift_segment'].value_counts().reset_index()
        segment_counts.columns = ['Segment', 'Count']
        fig = px.pie(segment_counts, values='Count', names='Segment', title='Uplift Segments')
        st.plotly_chart(fig, use_container_width=True)
        
    st.subheader("Uplift by Decile (Observed Differences)")
    st.markdown("Decile 1 = Customers with the highest predicted uplift. We expect the actual observed lift (Treatment Rate - Control Rate) to be highest in Decile 1 and lowest in Decile 10.")
    decile_results = evaluate_deciles(df)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=decile_results['decile'], y=decile_results['treatment_rate'], name='Treatment Conv. Rate', offsetgroup=0))
    fig.add_trace(go.Bar(x=decile_results['decile'], y=decile_results['control_rate'], name='Control Conv. Rate', offsetgroup=1))
    fig.add_trace(go.Scatter(x=decile_results['decile'], y=decile_results['actual_lift'], name='Observed Incremental Lift', mode='lines+markers', yaxis='y2', line=dict(color='black', width=3)))
    
    fig.update_layout(
        title='Observed Treatment vs Control Conversion by Predicted Uplift Decile', 
        xaxis_title='Decile (1 = Highest Predicted Uplift)', 
        yaxis_title='Conversion Rate',
        yaxis2=dict(title='Observed Lift', overlaying='y', side='right'),
        barmode='group'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("View Decile Data Table"):
        st.dataframe(decile_results.style.format({'treatment_rate': '{:.2%}', 'control_rate': '{:.2%}', 'actual_lift': '{:.2%}', 'predicted_uplift': '{:.2%}'}))
    
    st.subheader("Qini Curve (Incremental Gains)")
    x_axis, qini, random_line = qini_curve_data(df)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_axis, y=qini, mode='lines', name='Uplift Model'))
    fig.add_trace(go.Scatter(x=x_axis, y=random_line, mode='lines', name='Random Targeting', line=dict(dash='dash')))
    fig.update_layout(title='Qini Curve', xaxis_title='Targeted Population (%)', yaxis_title='Incremental Conversions')
    st.plotly_chart(fig, use_container_width=True)
    
    if fi_df is not None:
        st.markdown("---")
        st.subheader("Explainable AI: What drives uplift?")
        st.markdown("This chart shows the **Predictive Feature Importance** (via Permutation Importance) extracted from the T-Learner models. It highlights which customer attributes most strongly influence the underlying purchase probability models that determine the Individual Treatment Effect (ITE).")
        
        fig_fi = px.bar(
            fi_df.sort_values(by='Importance', ascending=True), 
            x='Importance', 
            y='Feature', 
            orientation='h',
            title='Feature Importance (Average of Treatment & Control Models)'
        )
        st.plotly_chart(fig_fi, use_container_width=True)
        
        st.info("**Interpretation Example:** If `discount_usage` has the highest importance, it means a customer's historical response to discounts is the most powerful predictor of whether this new campaign will successfully persuade them.")

# --- Tab 4: Customer Targeting ---
with tabs[3]:
    st.header("Customer Targeting Engine")
    
    col1, col2 = st.columns(2)
    with col1:
        min_uplift = st.slider("Minimum Expected Uplift (%)", min_value=-10.0, max_value=50.0, value=2.0, step=0.5) / 100.0
    with col2:
        segment_filter = st.multiselect("Select Segments to Target", 
                                        options=df['uplift_segment'].unique(),
                                        default=['Persuadables'])
                                        
    target_df = df[(df['pred_ite'] >= min_uplift) & (df['uplift_segment'].isin(segment_filter))]
    
    st.write(f"**Target Audience Size:** {len(target_df)} customers ({len(target_df)/len(df)*100:.1f}% of total)")
    
    export_df = target_df[['customer_id', 'uplift_segment', 'pred_prob_t1', 'pred_prob_t0', 'pred_ite', 'age', 'income', 'historical_spend']].sort_values(by='pred_ite', ascending=False)
    
    st.dataframe(export_df.head(100))
    
    csv = export_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Targeting List (CSV)",
        data=csv,
        file_name='marketing_targeting_export.csv',
        mime='text/csv',
    )

# --- Tab 5: ROI Simulator ---
with tabs[4]:
    st.header("ROI Targeting Simulator")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        target_pct = st.slider("Target % of Customers", 5, 100, 20, step=5)
    with col2:
        cost_per_cust = st.number_input("Campaign Cost per Customer ($)", value=2.0)
    with col3:
        rev_per_conv = st.number_input("Avg Revenue per Conversion ($)", value=100.0)
        
    roi_results = simulate_roi(
        df, target_pct, cost_per_cust, rev_per_conv,
        uplift_col='pred_ite', prob_col='pred_baseline_prob'
    )
    
    if roi_results:
        res_df = pd.DataFrame(roi_results).T
        st.dataframe(res_df.style.format("{:,.2f}"))
        
        fig = px.bar(res_df.reset_index(), x='index', y='Profit', title='Expected Profit by Targeting Strategy',
                     labels={'index': 'Targeting Strategy', 'Profit': 'Profit ($)'}, color='index')
        st.plotly_chart(fig, use_container_width=True)

# --- Tab 6: Customer Explorer ---
with tabs[5]:
    st.header("Individual Customer Explorer (Explainability)")
    
    sample_customers = df.sample(50, random_state=42)['customer_id'].tolist()
    selected_customer = st.selectbox("Select a Customer ID", sample_customers)
    
    cust_data = df[df['customer_id'] == selected_customer].iloc[0]
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Customer Profile")
        st.write(f"**Age:** {cust_data['age']}")
        st.write(f"**Income:** ${cust_data['income']:,.0f}")
        st.write(f"**Tenure (months):** {cust_data['customer_tenure']}")
        st.write(f"**Previous Purchases:** {cust_data['previous_purchases']}")
        st.write(f"**Email Open Rate:** {cust_data['email_open_rate']:.2f}")
        
    with col2:
        st.subheader("Model Predictions")
        st.write(f"**Segment:** {cust_data['uplift_segment']}")
        st.metric("Probability (If Treated)", f"{cust_data['pred_prob_t1']*100:.1f}%")
        st.metric("Probability (If Not Treated)", f"{cust_data['pred_prob_t0']*100:.1f}%")
        st.metric("Estimated Uplift (ITE)", f"{cust_data['pred_ite']*100:.1f}%")
        
        if cust_data['uplift_segment'] == 'Persuadables':
            st.success("Recommendation: **TARGET**. High expected incremental impact.")
        elif cust_data['uplift_segment'] == 'Sure Things':
            st.warning("Recommendation: **DO NOT TARGET**. Likely to buy anyway. Save budget.")
        elif cust_data['uplift_segment'] == 'Lost Causes':
            st.warning("Recommendation: **DO NOT TARGET**. Unlikely to buy regardless of campaign.")
        else:
            st.error("Recommendation: **DO NOT TARGET**. Campaign may cause negative impact (annoyance).")

