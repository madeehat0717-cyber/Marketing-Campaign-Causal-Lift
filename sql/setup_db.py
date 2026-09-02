import sqlite3
import pandas as pd
import os

def load_data_to_sqlite(csv_path, db_path='campaign.db', table_name='customers'):
    """
    Loads the processed data into an SQLite database for the SQL component.
    """
    df = pd.read_csv(csv_path)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    # Write data to sqlite (replace if exists)
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()
    print(f"Data loaded into SQLite database: {db_path}")

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(__file__))
    csv_path = os.path.join(base_dir, 'data', 'processed', 'scored_data.csv') # Assume pipeline saves this
    db_path = os.path.join(base_dir, 'data', 'campaign.db')
    
    if os.path.exists(csv_path):
        load_data_to_sqlite(csv_path, db_path)
    else:
        print(f"Could not find {csv_path}. Run the full pipeline first.")
