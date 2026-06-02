import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import yaml

def load_config(config_path="config.yaml"):
    """Loads the YAML configuration file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def preprocess_data(df, config):
    """Cleans, encodes, and scales the raw dataframe based exactly on EDA."""
    print("Starting data cleaning process...")
    
    # 1. Drop missing values AND duplicates
    df_clean = df.dropna().copy()
    df_clean = df_clean.drop_duplicates()
    
    # 2. Filter outliers
    t_low = config['preprocessing']['temp_lower_bound']
    t_high = config['preprocessing']['temp_upper_bound']
    h_low = config['preprocessing']['humidity_lower_bound']
    h_high = config['preprocessing']['humidity_upper_bound']
    
    df_clean = df_clean[
        (df_clean['Temperature'] >= t_low) & (df_clean['Temperature'] <= t_high) &
        (df_clean['Humidity'] >= h_low) & (df_clean['Humidity'] <= h_high)
    ]
    
    # 3. Drop useless columns
    if 'Session ID' in df_clean.columns:
        df_clean = df_clean.drop(columns=['Session ID'])
        
    # 4. Standardize Target Variable (Activity Level)
    activity_map = {
        'Low Activity': 0, 'Low_Activity': 0, 'LowActivity': 0, 
        'Moderate Activity': 1, 'Moderate_Activity': 1, 'ModerateActivity': 1, 
        'High Activity': 2, 'High_Activity': 2, 'HighActivity': 2
    }
    if 'Activity Level' in df_clean.columns:
        df_clean['Activity Level'] = df_clean['Activity Level'].astype(str).str.strip()
        df_clean['Activity Level'] = df_clean['Activity Level'].map(activity_map)
        df_clean = df_clean.dropna(subset=['Activity Level'])
        df_clean['Activity Level'] = df_clean['Activity Level'].astype(int)
        
    # 5. Encode Categorical Environmental Variables (UPDATED MAPPINGS!)
    light_map = {
        'very_dim': 0, 'Very_Dim': 0, 'Very Dim': 0, 'very dim': 0,
        'dim': 1, 'Dim': 1,
        'moderate': 2, 'Moderate': 2,
        'bright': 3, 'Bright': 3,
        'very_bright': 4, 'Very_Bright': 4, 'Very Bright': 4, 'very bright': 4
    }
    if 'Ambient Light Level' in df_clean.columns:
        df_clean['Ambient Light Level'] = df_clean['Ambient Light Level'].astype(str).str.strip()
        df_clean['Ambient Light Level'] = df_clean['Ambient Light Level'].map(light_map)
        
    time_map = {
        'morning': 0, 'Morning': 0,
        'afternoon': 1, 'Afternoon': 1,
        'evening': 2, 'Evening': 2,
        'night': 3, 'Night': 3
    }
    if 'Time of Day' in df_clean.columns:
        df_clean['Time of Day'] = df_clean['Time of Day'].astype(str).str.strip()
        df_clean['Time of Day'] = df_clean['Time of Day'].map(time_map)
        
    # One-Hot Encode HVAC Mode
    if 'HVAC Operation Mode' in df_clean.columns:
        df_clean = pd.get_dummies(df_clean, columns=['HVAC Operation Mode'], drop_first=True)
        
    # 6. Scale Specific Numerical Features
    min_max_cols = ['CO2_InfraredSensor']
    std_cols = ['CO2_ElectroChemicalSensor', 'MetalOxideSensor_Unit1', 'MetalOxideSensor_Unit2', 'CO_GasSensor']
    
    min_max_scaler = MinMaxScaler()
    std_scaler = StandardScaler()
    
    for col in min_max_cols:
        if col in df_clean.columns:
            df_clean[col] = min_max_scaler.fit_transform(df_clean[[col]])
            
    for col in std_cols:
        if col in df_clean.columns:
            df_clean[col] = std_scaler.fit_transform(df_clean[[col]])
            
    # --- THE ULTIMATE SAFETY NET ---
    df_clean = df_clean.dropna()
            
    print(f"Final dataset shape ready for training: {df_clean.shape}")
    return df_clean

if __name__ == "__main__":
    import sqlite3
    
    print("Testing Data Cleaning script independently...")
    config = load_config()
    
    conn = sqlite3.connect(config['data']['db_path'])
    df_raw = pd.read_sql_query(f"SELECT * FROM {config['data']['table_name']}", conn)
    conn.close()
    
    clean_df = preprocess_data(df_raw, config)
    print(clean_df[['Temperature', 'Humidity', 'Activity Level']].head())
