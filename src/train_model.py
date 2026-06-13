import os
import sys
import yaml
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from ingest_data import ingest_from_db
from clean_data import preprocess_data

# Import SMOTE for handling class imbalance
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    print("Warning: 'imbalanced-learn' is not installed. SMOTE will be skipped.")

# Add current directory to path so can import other scripts
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def load_config(config_path="config.yaml"):
    """Loads the YAML configuration file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def execute_training_pipeline():
    """ load the raw dataset and clean it with clean_data.py

    Apply train test split at 0.2 test_size and 42 random_state and loop thru
    the selected model based on config file for 3 algorithm models which is RandomForest, GradientBoosting,
    DecisionTree. Afterwards saved the model into folder for evaluation use.
    
    """
    print("=== STARTING TRAINING PIPELINE ===")
    config = load_config()

    # Ingest Raw Data
    print("\nPulling raw data using ingest_data.py...")
    df_raw = ingest_from_db(config['data']['db_path'], config['data']['table_name'])

    # Use Clean_data.py
    print("\nCleaning data using clean_data.py...")
    df_clean = preprocess_data(df_raw, config)

    # Split Features (X) and Target (y)
    print("\nSplitting data for training...")
    X = df_clean.drop(columns=['Activity Level'])
    y = df_clean['Activity Level']

    test_size = config['model'].get('test_size', 0.2)
    random_state = config['model'].get('random_state', 42)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Data Augmentation: SMOTE
    print("\nApplying SMOTE to balance the training data...")
    try:
        smote = SMOTE(random_state=random_state)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        print(f"Original training shape: {X_train.shape} | Resampled shape: {X_train_resampled.shape}")
    except NameError:
        print("SMOTE library not found. Do it with un-resampled data.")
        X_train_resampled, y_train_resampled = X_train, y_train

    # Initialize Model Dynamically with GridSearchCV
    selected_model = config['model']['algorithm']
    print(f"\nInitializing {selected_model} algorithm...")

    if selected_model == "RandomForest":
        base_model = RandomForestClassifier(random_state=random_state, class_weight='balanced')
        # Hyperparameter Tuning Grid
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20]
        }
        # Using recall_macro to prevent false nagatives
        model = GridSearchCV(base_model, param_grid, cv=3, scoring='recall_macro', n_jobs=-1)

    elif selected_model == "GradientBoosting":
        base_model = GradientBoostingClassifier(random_state=random_state)
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5]
        }
        model = GridSearchCV(base_model, param_grid, cv=3, scoring='recall_macro', n_jobs=-1)

    elif selected_model == "DecisionTree":
        model = DecisionTreeClassifier(random_state=random_state, class_weight='balanced')

    else:
        raise ValueError(f"Algorithm '{selected_model}' is not supported! Check config.yaml.")

    # Train the Model
    print(f"\nTraining the {selected_model} model...")
    model.fit(X_train_resampled, y_train_resampled)

    # Extract the best model if GridSearchCV was used
    if hasattr(model, 'best_params_'):
        print(f"Best parameters found by GridSearchCV: {model.best_params_}")
        final_model = model.best_estimator_
    else:
        print("Model training complete!")
        final_model = model

    # Save the Model
    model_path = f"saved_model/{selected_model}_model.pkl"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(final_model, f)

    # Save training and testing Data for evaluation.py
    print("\nSaving train and test data to CSV for evaluation/debugging...")
    os.makedirs("data", exist_ok=True) 
    X_train.to_csv("data/X_train.csv", index=False)
    y_train.to_csv("data/y_train.csv", index=False)
    X_test.to_csv("data/X_test.csv", index=False)
    y_test.to_csv("data/y_test.csv", index=False)

    print(f"\nModel saved to {model_path}")
    return final_model, X_train, X_test, y_train, y_test

if __name__ == "__main__":
    execute_training_pipeline()