import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import yaml
import sqlite3

def load_config(config_path="config.yaml"):
    """
    Loads the YAML configuration file. 
    If the file is not found, it falls back to this internal dictionary.
    """
    try:
        with open(config_path, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print("Warning: config.yaml not found. Using internal configuration.")
        return {
            'preprocessing': {
                'temp_lower_bound': 10,
                'temp_upper_bound': 50,
                'humidity_lower_bound': 0,
                'humidity_upper_bound': 100
            },
            'data': {
                'db_path': 'gas_monitoring.db',
                'table_name': 'gas_monitoring'
            }
        }

def impute_numerical_column(df, col_name, is_discrete=False):
    """
    Custom Stochastic Bounded Gaussian Imputation.
    If is_discrete=True, the generated value is rounded to the nearest whole integer.
    """
    rows_to_drop = []
    missing_indices = df[df[col_name].isna()].index
    
    for idx in missing_indices:
        session_id = df.loc[idx, 'Session ID']
        time_of_day = df.loc[idx, 'Time of Day']
        
        # Filter dataframe to the specific context group
        group_mask = (df['Session ID'] == session_id) & \
                     (df['Time of Day'] == time_of_day) & \
                     (df[col_name].notna())
        group_data = df.loc[group_mask, col_name]
        
        # Ensure we have enough data to calculate a valid standard deviation
        if len(group_data) > 1 and group_data.std() > 0:
            g_mean = group_data.mean()
            g_std = group_data.std()
            g_min = group_data.min()
            g_max = group_data.max()
            
            # Generate bounded random number from a normal distribution
            random_val = np.random.normal(loc=g_mean, scale=g_std)
            random_val = np.clip(random_val, g_min, g_max)
            
            # Round to nearest whole number if it is a discrete category (like 0,1,2,3)
            if is_discrete:
                random_val = np.round(random_val)
                
            df.loc[idx, col_name] = random_val
        else:
            # Fallback: Mark row to be dropped if group data is insufficient
            rows_to_drop.append(idx)
            
    # Safely drop rows outside of the iteration loop
    if rows_to_drop:
        df = df.drop(index=rows_to_drop)
        
    return df

def clean_and_encode(df, config):
    """Apply data cleaning, out-of-bounds nullification, and categorical encoding."""
    df_clean = df.copy()
    
    # 1. Temperature Kelvin Glitch Fix
    if 'Temperature' in df_clean.columns:
        df_clean.loc[df_clean['Temperature'] > 250, 'Temperature'] = df_clean.loc[df_clean['Temperature'] > 250, 'Temperature'] - 273.15
        
        # Apply Config Bounds: Turn out-of-bounds Temperatures into NaN for imputation
        t_low = config['preprocessing']['temp_lower_bound']
        t_high = config['preprocessing']['temp_upper_bound']
        df_clean.loc[(df_clean['Temperature'] < t_low) | (df_clean['Temperature'] > t_high), 'Temperature'] = np.nan

    # 2. Humidity Bounds: Turn out-of-bounds Humidities into NaN for imputation
    if 'Humidity' in df_clean.columns:
        h_low = config['preprocessing']['humidity_lower_bound']
        h_high = config['preprocessing']['humidity_upper_bound']
        df_clean.loc[(df_clean['Humidity'] < h_low) | (df_clean['Humidity'] > h_high), 'Humidity'] = np.nan

    # 3. Time of Day Mapping
    time_map = {
        'morning': 1, 'Morning': 1,
        'afternoon': 2, 'Afternoon': 2,
        'evening': 3, 'Evening': 3,
        'night': 4, 'Night': 4
    }
    if 'Time of Day' in df_clean.columns:
        df_clean['Time of Day'] = df_clean['Time of Day'].astype(str).str.strip()
        df_clean['Time of Day'] = df_clean['Time of Day'].map(time_map)

    # 4. Ambient Light Level Ordinal Encoding
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
        
    # 5. Activity Level Target Variable Consolidation & Ordinal Encoding
    activity_map = {
        'Low Activity': 0, 'Low_Activity': 0, 'LowActivity': 0, 
        'Moderate Activity': 1, 'Moderate_Activity': 1, 'ModerateActivity': 1, 
        'High Activity': 2, 'High_Activity': 2, 'HighActivity': 2
    }
    if 'Activity Level' in df_clean.columns:
        df_clean['Activity Level'] = df_clean['Activity Level'].astype(str).str.strip()
        df_clean['Activity Level'] = df_clean['Activity Level'].map(activity_map)
        
    # 6. HVAC Operation Mode Formatting & One-Hot Encoding
    if 'HVAC Operation Mode' in df_clean.columns:
        df_clean['HVAC Operation Mode'] = df_clean['HVAC Operation Mode'].str.upper()
        df_clean = pd.get_dummies(df_clean, columns=['HVAC Operation Mode'], drop_first=True, dtype=int)
        
    return df_clean

def scale_features(df):
    """Standardize the environmental sensor features."""
    std_cols = [
        'CO2_ElectroChemicalSensor', 'MetalOxideSensor_Unit1', 
        'MetalOxideSensor_Unit2', 'MetalOxideSensor_Unit3', 
        'MetalOxideSensor_Unit4', 'CO_GasSensor', 'CO2_InfraredSensor'
    ]
            
    std_scaler = StandardScaler()
    for col in std_cols:
        if col in df.columns:
            df[col] = std_scaler.fit_transform(df[[col]])
            
    return df

def preprocess_data(df, config):
    """Main pipeline execution function."""
    print("Starting data cleaning process...")
    
    # 1. Drop complete duplicates
    df = df.drop_duplicates()
    
    # 2. Clean, Map, and turn out-of-bounds to NaNs
    df = clean_and_encode(df, config)
    
    # 3. Execute Stochastic Imputation
    print("Executing stochastic imputation pipeline...")
    # Added 'Temperature' since we nullified out-of-bounds temp rows
    imputation_columns = ['Temperature', 'Humidity', 'MetalOxideSensor_Unit2', 'CO_GasSensor', 'Ambient Light Level']
    
    for col in imputation_columns:
        if col in df.columns:
            # Check if this column needs to be rounded to a discrete integer
            is_discrete = col in ['CO_GasSensor', 'Ambient Light Level']
            
            df = impute_numerical_column(df, col, is_discrete=is_discrete)
            print(f" - Imputation for '{col}' complete.")
            
    # 4. Ultimate Safety Net: Drop any remaining rows with NaNs (e.g., unmappable targets)
    df = df.dropna()

    # 5. Scale Features
    print("Scaling sensor features...")
    df = scale_features(df)
    
    # 6. Drop 'Session ID' now that imputation is fully finished
    if 'Session ID' in df.columns:
        df = df.drop(columns=['Session ID'])
    
    print(f"Final dataset shape ready for training: {df.shape}")
    return df

if __name__ == "__main__":
    print("Testing Data Cleaning script independently...")
    
    # Load config (reads file or uses default dictionary)
    config = load_config()
    
    # Connect to database using path from config
    db_path = config['data']['db_path']
    table_name = config['data']['table_name']
    
    conn = sqlite3.connect(db_path)
    df_raw = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()
    
    # Run the entire pipeline in memory
    clean_df = preprocess_data(df_raw, config)
    
    print("Test successful! Pipeline executed cleanly.")
    # Show preview to confirm rounding worked on CO_GasSensor
    print(clean_df[['Temperature', 'Humidity', 'CO_GasSensor', 'Activity Level']].head())
