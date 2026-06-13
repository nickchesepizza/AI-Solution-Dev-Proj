"""
Model Evaluation Module

This script handles loading the configuration, loading the pre-saved test data split, 
loading a trained machine learning model, and outputting performance metrics 
and feature importances.
"""

import pandas as pd
import yaml
import pickle 
import os 
import sys 
from sklearn.metrics import classification_report, confusion_matrix

# Function to load configuration settings from a given YAML file
def load_config(config_path="config.yaml"):
    """Loads the YAML configuration file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def evaluate_model():
    """Loads the trained model and evaluates it on the saved test csv

    This function coordinates the ingestion and preprocessing of data, 
    splits it to do the test set, loads the machine learning 
    model from saved ones, and prints evaluation metrics ( like classification report and 
    confusion matrix) together with key feature importances.
    
    """
    print("=== STARTING EVALUATION PIPELINE ===")

    config = load_config()

    # 1. Load the pre-saved test data
    if not os.path.exists("data/X_test.csv") or not os.path.exists("data/y_test.csv"):
        print("Error: Test data not found. Please run train_model.py first!")
        return
        
    print("\nLoading pre-cleaned test data...")
    X_test = pd.read_csv("data/X_test.csv")
    y_test = pd.read_csv("data/y_test.csv")['Activity Level'] # Extract the target column

    # 2. Load the specific model requested in config.yaml
    algo = config['model']['algorithm']
    model_path = f"saved_model/{algo}_model.pkl"
    
    if not os.path.exists(model_path):
        print(f"Error: Could not find {model_path}. Did you train this model yet?")
        return

    print(f"\nLoading {algo} model from {model_path}...")
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
    print("\nKEY FEATURE IMPORTANCE:")
    
    if hasattr(model, 'feature_importances_'):
        # This handles Random Forest and Decision Tree
        importances = pd.Series(model.feature_importances_, index=X_test.columns)
        top_features = importances.sort_values(ascending=False).head(5)
        print(top_features)
        
    else:
        print(f"Note: {algo} does not have native feature_importances_.")
        
    print("\nEvaluation Complete!\n")

if __name__ == "__main__":
    evaluate_model()