-- SQL Component: Analytical Queries

-- 1. Customer Summary: Total customers and general metrics
SELECT 
    COUNT(customer_id) AS total_customers,
    ROUND(AVG(age), 1) AS avg_age,
    ROUND(AVG(income), 2) AS avg_income,
    ROUND(AVG(average_order_value), 2) AS avg_order_value
FROM customers;

-- 2. Treatment vs Control Counts and Conversion Rates
SELECT 
    treatment,
    COUNT(customer_id) AS customers,
    SUM(purchase_outcome) AS conversions,
    ROUND(CAST(SUM(purchase_outcome) AS FLOAT) / COUNT(customer_id) * 100, 2) AS conversion_rate_pct
FROM customers
GROUP BY treatment;

-- 3. Revenue by Campaign Assignment
SELECT 
    treatment,
    SUM(revenue) AS total_revenue,
    ROUND(AVG(revenue), 2) AS avg_revenue_per_customer
FROM customers
GROUP BY treatment;

-- 4. Revenue by Uplift Segment
SELECT 
    uplift_segment,
    COUNT(customer_id) AS segment_size,
    ROUND(AVG(pred_ite), 4) AS avg_estimated_uplift,
    ROUND(AVG(purchase_outcome), 4) AS actual_conversion_rate
FROM customers
GROUP BY uplift_segment
ORDER BY avg_estimated_uplift DESC;

-- 5. Top 10 Customers to Target (Highest Uplift)
SELECT 
    customer_id,
    age,
    income,
    pred_prob_t1 AS probability_with_treatment,
    pred_prob_t0 AS probability_without_treatment,
    pred_ite AS estimated_uplift
FROM customers
ORDER BY pred_ite DESC
LIMIT 10;

-- 6. High-Risk Customers (Sleeping Dogs / Do-Not-Disturb)
-- These are customers where the campaign is expected to decrease their purchase probability
SELECT 
    customer_id,
    pred_ite AS negative_uplift,
    pred_prob_t0 AS high_base_probability
FROM customers
WHERE pred_ite < 0
ORDER BY pred_ite ASC
LIMIT 10;

-- 7. Campaign Performance: Aggregate metrics
SELECT 
    SUM(CASE WHEN treatment = 1 THEN purchase_outcome ELSE 0 END) AS treatment_conversions,
    SUM(CASE WHEN treatment = 0 THEN purchase_outcome ELSE 0 END) AS control_conversions,
    (SELECT COUNT(*) FROM customers WHERE treatment = 1) AS treatment_size,
    (SELECT COUNT(*) FROM customers WHERE treatment = 0) AS control_size
FROM customers;
