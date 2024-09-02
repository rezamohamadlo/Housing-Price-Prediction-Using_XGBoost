# Melburne-Housing-Price-prediction
"A machine learning project that predicts Melbourne housing prices using XGBoost. The model is trained on key housing features like room count, distance from the city, and building area, with performance tuning for optimal accuracy."

Dataset
The dataset used in this project is the Melbourne Housing Snapshot from Kaggle. It contains various features related to housing in Melbourne, Australia, and the target variable is the price of the house.

Project Structure
melb_data.csv: The dataset file used for training and validation.
housing_price_prediction.py: The Python script that reads the data, processes it, trains the model, and evaluates its performance.
Requirements
The project requires the following Python packages:

pandas
scikit-learn
xgboost
You can install the required packages using pip:

bash
Copy code
pip install pandas scikit-learn xgboost
How to Run
Load the Dataset: The dataset is loaded from a CSV file.

python
Copy code
data = pd.read_csv('/kaggle/input/melbourne-housing-snapshot/melb_data.csv')
Select Features and Target: The features (X) and target (y) are selected from the dataset.

python
Copy code
cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
X = data[cols_to_use]
y = data.Price
Split the Data: The data is split into training and validation sets.

python
Copy code
X_train, X_valid, y_train, y_valid = train_test_split(X, y)
Train the Model: The XGBoost model is trained on the training data.

python
Copy code
my_model = XGBRegressor()
my_model.fit(X_train, y_train)
Evaluate the Model: The model's performance is evaluated using Mean Absolute Error (MAE).

python
Copy code
predictions = my_model.predict(X_valid)
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))
Model Tuning: The model is tuned with different hyperparameters to improve performance.

Increasing the number of estimators to 500.
Adding early stopping with 5 rounds.
Adjusting the learning rate to 0.05 and increasing the number of estimators to 1000.
Using multiple threads for training by setting n_jobs=4.
python
Copy code
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)
my_model.fit(X_train, y_train, 
             early_stopping_rounds=5, 
             eval_set=[(X_valid, y_valid)], 
             verbose=False)
Results
The model's performance is measured using Mean Absolute Error (MAE). The tuning steps involve experimenting with different numbers of estimators, learning rates, and early stopping to find the optimal model configuration.

License
This project is licensed under the MIT License.

Acknowledgments
The dataset used in this project is provided by Kaggle.
The XGBoost library is used for training the model.

