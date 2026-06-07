"""
Model Evaluation Module

This script handles loading the configuration, regenerating the test data split, 
loading a trained machine learning model, and outputting performance metrics 
and feature importances.
"""

import pandas as pd
import yaml
import pickle #use it to load previously trained machine learning model from the saved ".pkl" file so we can do predictions
import os  #use it to get the path of the current script
import sys #use it together with os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Pull functions from your other files
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ingest_data import ingest_from_db
from clean_data import preprocess_data

# Function to load configuration settings from a given YAML file
def load_config(config_path="config.yaml"):
    """Loads the YAML configuration file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def evaluate_model():
    """Loads the trained model and evaluates it on test data.

    This function coordinates the ingestion and preprocessing of data, 
    splits it to do the test set, loads the machine learning 
    model from saved ones, and prints evaluation metrics ( like classification report and 
    confusion matrix) together with key feature importances."""
    print("STARTING EVALUATION PIPELINE")

    config = load_config()

    # 1. Recreate the exact same test data
    df_raw = ingest_from_db(config['data']['db_path'], config['data']['table_name'])
    df_clean = preprocess_data(df_raw, config)

    X = df_clean.drop(columns=['Activity Level'])
    y = df_clean['Activity Level']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config['model']['test_size'], 
        random_state=config['model']['random_state'], 
        stratify=y
    )

    # 2. Load the specific model requested in config.yaml
    algo = config['model']['algorithm']
    model_path = f"saved_model/{algo}_model.pkl"
    
    if not os.path.exists(model_path):
        print(f"Error: Could not find {model_path}. Did you train this model yet?")
        return

    print(f"\n Loading {algo} model from {model_path}...")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # 3. Make Predictions
    print(f"Testing the model on {len(X_test)} hidden rows...")
    y_pred = model.predict(X_test)

    # 4. Print Metrics for the Report
    print("\n" + "="*50)
    print(f"{algo.upper()} EVALUATION RESULTS")
    print("="*50)
    
    print("\nCLASSIFICATION REPORT:")
    # Target names to match 0=Low, 1=Moderate, 2=High mapping
    print(classification_report(y_test, y_pred, target_names=['Low (0)', 'Moderate (1)', 'High (2)']))
    
    print("CONFUSION MATRIX:")
    print(confusion_matrix(y_test, y_pred))

    # 5. Extract Feature Importance 
    print("\n KEY FEATURE IMPORTANCE:")
    
    if hasattr(model, 'feature_importances_'):
        # This handles Random Forest and Decision Tree
        importances = pd.Series(model.feature_importances_, index=X_test.columns)
        top_features = importances.sort_values(ascending=False).head(5)
        print(top_features)
        
    elif hasattr(model, 'coef_'):
        # This handles Logistic Regression!
        import numpy as np
        # Because there are 3 classes (0,1,2), we calculate the absolute average weight of each sensor
        importances = pd.Series(np.mean(np.abs(model.coef_), axis=0), index=X_test.columns)
        top_features = importances.sort_values(ascending=False).head(5)
        print(top_features)
        
    else:
        print(f"Note: {algo} does not have native feature_importances_.")
        
    print("\nEvaluation Complete!\n")

if __name__ == "__main__":
    evaluate_model()