import sqlite3
import pandas as pd
import yaml

def load_config(config_path="config.yaml"):
    """Loads the YAML configuration file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def ingest_from_db(db_path, table_name):
    """Connects to the SQLite DB and extracts the data into a Pandas DataFrame."""
    print(f"Connecting to database at: {db_path}...")
    
    # Establish connection using sqlite3
    conn = sqlite3.connect(db_path)
    
    # Query the database
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql_query(query, conn)
    
    # Close the connection
    conn.close()
    
    print(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns.")
    return df

if __name__ == "__main__":
    # test this specific script independently
    print("Testing Data Ingestion...")
    config = load_config()
    df = ingest_from_db(config['data']['db_path'], config['data']['table_name'])
    print(df.head())