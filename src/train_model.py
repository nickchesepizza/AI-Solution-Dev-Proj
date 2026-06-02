import pandas as pd
import yaml
import pickle
import os
import sys
from sklearn.model_selection import train_test_split

def load_config(config_path="config.yaml"):
    """Loads the YAML configuration file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def train_and_save_model(df, config):
    """Splits data, dynamically trains the selected model, and saves it."""
    print("Starting model training phase...")

    # 1. Separate Features (X) and Target (y)
    X = df.drop(columns=['Activity Level'])
    y = df['Activity Level']

    # 2. Split into Training and Testing Sets
    test_size = config['model']['test_size']
    random_state = config['model']['random_state']
    
    print(f"Splitting data (Test Size: {test_size * 100}%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 3. Initialize the Model dynamically based on config.yaml
    selected_model = config['model']['algorithm']
    print(f"Initializing {selected_model} algorithm...")
    
    if selected_model == "RandomForest":
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import GridSearchCV

        # 1. Base model
        rf = RandomForestClassifier(random_state=random_state, class_weight='balanced')
        
        # 2. Hyperparameter Grid to tune
        param_grid = {
            'n_estimators': [200, 500, 100],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'max_features': ['sqrt', 'log2']
        }
        
        # 3. GridSearchCV to find the best combination
        print("Starting Hyperparameter Tuning for Random Forest...")
        model = GridSearchCV(rf, param_grid, cv=3, scoring='f1_weighted', verbose=1)
        model.fit(X_train, y_train)
        
        print(f"Best parameters found: {model.best_params_}")
        model = model.best_estimator_
        
    elif selected_model == "DecisionTree":
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier(random_state=random_state, class_weight='balanced')
        
    elif selected_model == "GradientBoosting":
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import GridSearchCV

        # 1. Define the base model
        gb = GradientBoostingClassifier(random_state=random_state)
        
        # 2. Define the parameters you want to tune
        # This tells the computer: "Try every combination of these settings"
        param_grid = {
            'n_estimators': [100, 200, 500],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5]
        }
        
        # 3. Setup GridSearchCV
        # cv=3 means it will split your training data into 3 parts to test each setting
        print("Starting Hyperparameter Tuning with GridSearchCV...")
        model = GridSearchCV(gb, param_grid, cv=3, scoring='f1_weighted', verbose=1)
        
        # 4. Train it
        # The model will automatically test all combinations and pick the best one
        model.fit(X_train, y_train)
        
        print(f"Best parameters found: {model.best_params_}")
        
        # 5. Set 'model' to the best version found
        model = model.best_estimator_
        
    else:
        raise ValueError(f"Algorithm '{selected_model}' is not supported! Please check config.yaml.")

    # 4. Train (Fit) the Model
    print(f"Training the {selected_model} model on {len(X_train)} rows...")
    model.fit(X_train, y_train)
    print("Model training complete!")

    # 5. Save the trained model
    model_path = f"saved_model/{selected_model}_model.pkl"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
        
    print(f"Model successfully saved to {model_path}")
    return model, X_train, X_test, y_train, y_test

if __name__ == "__main__":
    import sqlite3
    print("Testing Model Training Script...")
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from ingest_data import ingest_from_db
    from clean_data import preprocess_data
    
    config = load_config()
    df_raw = ingest_from_db(config['data']['db_path'], config['data']['table_name'])
    df_clean = preprocess_data(df_raw, config)
    train_and_save_model(df_clean, config)