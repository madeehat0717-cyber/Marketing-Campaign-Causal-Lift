# Notebook: EDA and Causal Analysis

This is a placeholder for `notebooks/01_eda_and_causal_analysis.ipynb`.

Run the Streamlit app to view all EDA and Causal Analysis interactively.

To explore the data via Python, you can use:

```python
import pandas as pd
import sqlite3

conn = sqlite3.connect('../data/campaign.db')
df = pd.read_sql('SELECT * FROM customers', conn)

print(df.describe())
```
