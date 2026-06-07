## Group Name: Team 1
## Members: Danish, Zaw Moon, Daniel
Zaw Moon → injest_data.py and evaluate.py\
Danish → clean_data.py\
Daniel → train_model.py

## Pipeline Instruction

This project uses a containerized Docker environment

### 1. Start the Docker Development Environment
Before running the pipeline, you need to build and start the container. Open your terminal, navigate to the a folder for this project, and run:
```bash
git pull origin main

docker compose up --build -d

# Do this if you have already build the container
docker compose up -d
```

### 2. Running the Machine Learning Pipeline
Once the environment is running, you can execute the pipeline directly inside the terminal
```bash
# To train the model
'''NOTE: to change the algorithm for training, navigate to config.yaml and change
        the algorithm with this selected 3 algorithm: RandomForest, DecisionTree, GradientBoosting
'''
docker exec -it gas_monitoring_env python src/train_model.py

# To evaluate the model
docker exec -it gas_monitoring_env python src/evaluate.py
```

### 3. Shutting Down the Environment
When you are finished testing the pipeline, cleanly shut down the docker container and free up your system resources, by running:
```bash
docker compose down

```

## Key Findings of EDA

## Features that are engineered (clean data features)

## Choice of Models
Model used was DecisionTree, RandomForest, GradientBoosting.
DecisionTree → DecisionTree is fast to train and can copy human logic like if CO2 is high and humidity is rising, trigger alarm

RandomForest → RandomForest can train hundreds of individual trees on random subset and have a vote on the final prediction. like each tree see different subset and is able to decide which is best.

GradientBoosting → GradientBoosting instead of hundreds together at once, it build sequentially. Like tree 1 see this but tree 2 see that and correct it and more.

### Tuning Methods
GridSearchCV → using GridSearchCV to test combination of hyperparameters like max_depth and n_estimators so instead of guessing, it can find the optimal configuration within the grid.

K-Fold Cross Validation → ensuring the model dont memorize the training data, applying 3 fold cross validation during training, by rotating the train and validation multiple times to ensure the final model is generalize on new unseen data.

## Choice of Metrics
Since our project focus mainly on healthcare monitoring system, the suitable choice of metrice will be recall since it minimizes false negatives to ensure no critical events go undetected.

