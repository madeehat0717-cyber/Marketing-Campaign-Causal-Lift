# Architecture Diagram

This document outlines the high-level architecture and data flow of the Marketing Campaign Causal Lift platform.

```mermaid
graph TD
    A[Synthetic Data Generation] --> B[Raw Data]
    B --> C[Data Cleaning & Validation]
    C --> D[Feature Engineering]
    D --> E[Processed Features]
    
    E --> F[Causal Analysis ATE]
    
    E --> G[Model Training: Control Model]
    E --> H[Model Training: Treatment Model]
    
    G --> I[Uplift Scoring ITE]
    H --> I
    
    I --> J[Customer Segmentation]
    J --> K[Scored Data]
    
    K --> L[SQLite Database]
    L --> M[SQL Queries & Analytics]
    
    K --> N[Evaluation Metrics Qini/Decile]
    K --> O[ROI Simulator]
    
    N --> P[Streamlit Dashboard]
    O --> P
    M --> P
    F --> P
```

## Layers
1. **Data Layer:** `src/data_generation.py`, `src/data_cleaning.py`, `src/feature_engineering.py`
2. **Modeling Layer:** `src/causal_analysis.py`, `src/uplift_model.py`
3. **Business Logic Layer:** `src/segmentation.py`, `src/evaluation.py`, `src/roi_simulator.py`
4. **Data Storage Layer:** `sql/setup_db.py`, SQLite `campaign.db`
5. **Presentation Layer:** `app/streamlit_app.py`
