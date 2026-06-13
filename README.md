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
- **Wrong Temperatures:** Some temperature numbers were way too high because the sensors recorded them in Kelvin instead of standard Celsius.
- **Impossible Humidity:** We found humidity readings that were impossible (like below 0% or above 100%).
- **Missing Data:** Several sensors randomly stopped recording, leaving blank spaces in our data.
- **Good Baselines:** The "Time of Day" and "Session ID" columns were perfectly complete with no missing data. This means we can rely on them to figure out what the missing sensor data should be.
  
## Features that are engineered (clean data features)
- **Words to Numbers:** We changed the "Time of Day" words into simple numbers (Morning = 1, Afternoon = 2, Evening = 3, Night = 4) because machine learning models work best with numbers.
- **Fixing Temperatures:** We corrected the super high temperature readings by using a math formula to change them from Kelvin back to normal Celsius.
- **Clearing Bad Humidity:** We deleted the impossible humidity numbers (under 0 or over 100) and temporarily made them blank so we could fix them properly in the next step.
- **Data Imputation (Filling in the Blanks):** To fix the missing data for humidity and gas sensors, we did not just guess randomly. We grouped the data by "Time of Day" and "Session ID", looked at what the normal spread of numbers was for that specific time, and used that normal range to safely fill in the blanks.
- **Scaling Data:** Some sensors spit out very big numbers and others spit out very small numbers. We scaled the "Metal Oxide Sensor" columns so their numbers are all on the same playing field, which helps the model learn faster without getting confused.
- **Rounding Numbers:** We rounded the CO gas sensor numbers to the nearest whole number to keep the readings simple and clean.
- **Splitting Air Conditioning Modes:** The air conditioning (HVAC) column had different words in it (like "Off", "Eco", "Heating"). We split these into separate Yes/No columns (e.g., a column just for "Is Heating On? Yes or No") to make it easier for the AI to read.

## Choice of Models
Model used was DecisionTree, RandomForest, GradientBoosting.
- **DecisionTree** → DecisionTree is fast to train and can copy human logic like if CO2 is high and humidity is rising, trigger alarm
- **RandomForest** → RandomForest can train hundreds of individual trees on random subset and have a vote on the final prediction. like each tree see different subset and is able to decide which is best.
- **GradientBoosting** → GradientBoosting instead of hundreds together at once, it build sequentially. Like tree 1 see this but tree 2 see that and correct it and more.

### Tuning Methods
- **GridSearchCV** → using GridSearchCV to test combination of hyperparameters like max_depth and n_estimators so instead of guessing, it can find the optimal configuration within the grid.
- **K-Fold Cross Validation** → ensuring the model don't memorize the training data, applying 3 fold cross validation during training, by rotating the train and validation multiple times to ensure the final model is generalize on new unseen data

## Choice of Metrics
- **Recall** → Since our project focus mainly on healthcare monitoring system, the suitable choice of metrice will be recall since it minimizes false negatives to ensure no critical events go undetected.
- **Confusion Matrix** → to see how many critical events does the model predict wrongly in the test dataset.
- **Key feature importance** → to know which feature are the main contributor to detect the activity.

