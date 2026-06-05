import os
import sys
import yaml
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV

# Import SMOTE for handling class imbalance
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    print("Warning: 'imbalanced-learn' is not installed. SMOTE will be skipped.")

# Add current directory to path so we can import your other scripts
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ingest_data import ingest_from_db
from clean_data import preprocess_data

def load_config(config_path="config.yaml"):
    """Loads the YAML configuration file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def execute_training_pipeline():
    print("=== STARTING TRAINING PIPELINE ===")
    config = load_config()

    # [Step 1] Ingest Raw Data
    print("\n[Step 1] Pulling raw data using ingest_data.py...")
    df_raw = ingest_from_db(config['data']['db_path'], config['data']['table_name'])

    # [Step 2] Clean Data (Using your newly updated script)
    print("\n[Step 2] Cleaning data using clean_data.py...")
    df_clean = preprocess_data(df_raw, config)

    # [Step 3] Split Features (X) and Target (y)
    print("\n[Step 3] Splitting data for training...")
    X = df_clean.drop(columns=['Activity Level'])
    y = df_clean['Activity Level']

    test_size = config['model'].get('test_size', 0.2)
    random_state = config['model'].get('random_state', 42)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # [Step 4] Data Augmentation: SMOTE
    print("\n[Step 4] Applying SMOTE to balance the training data...")
    try:
        smote = SMOTE(random_state=random_state)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        print(f"Original training shape: {X_train.shape} | Resampled shape: {X_train_resampled.shape}")
    except NameError:
        print("SMOTE library not found. Proceeding with un-resampled data.")
        X_train_resampled, y_train_resampled = X_train, y_train

    # [Step 5] Initialize Model Dynamically with GridSearchCV
    selected_model = config['model']['algorithm']
    print(f"\n[Step 5] Initializing {selected_model} algorithm...")

    if selected_model == "RandomForest":
        from sklearn.ensemble import RandomForestClassifier
        base_model = RandomForestClassifier(random_state=random_state, class_weight='balanced')
        # Hyperparameter Tuning Grid
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20]
        }
        # recall_macro forces the model to care about the rare "High Activity" events
        model = GridSearchCV(base_model, param_grid, cv=3, scoring='recall_macro', n_jobs=-1)

    elif selected_model == "GradientBoosting":
        from sklearn.ensemble import GradientBoostingClassifier
        base_model = GradientBoostingClassifier(random_state=random_state)
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5]
        }
        model = GridSearchCV(base_model, param_grid, cv=3, scoring='recall_macro', n_jobs=-1)

    elif selected_model == "DecisionTree":
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier(random_state=random_state, class_weight='balanced')

    elif selected_model == "LogisticRegression":
        from sklearn.linear_model import LogisticRegression
        # max_iter increased to prevent convergence warnings with scaled data
        model = LogisticRegression(random_state=random_state, class_weight='balanced', max_iter=1000)

    else:
        raise ValueError(f"Algorithm '{selected_model}' is not supported! Check config.yaml.")

    # [Step 6] Train the Model
    print(f"\n[Step 6] Training the {selected_model} model...")
    model.fit(X_train_resampled, y_train_resampled)

    # Extract the best model if GridSearchCV was used
    if hasattr(model, 'best_params_'):
        print(f"Best parameters found by GridSearchCV: {model.best_params_}")
        final_model = model.best_estimator_
    else:
        print("Model training complete!")
        final_model = model

    # [Step 7] Save the Model
    model_path = f"saved_model/{selected_model}_model.pkl"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(final_model, f)

    print(f"\n✅ SUCCESS! Model saved perfectly to {model_path}")
    return final_model, X_train, X_test, y_train, y_test

if __name__ == "__main__":
    execute_training_pipeline()