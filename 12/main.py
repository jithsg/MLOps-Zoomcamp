import mlflow
import mlflow.sklearn
import numpy as np
import cloudpickle
import sklearn
import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

mlflow.set_tracking_uri("http://localhost:5000")    

mlflow.sklearn.autolog()

exp = mlflow.set_experiment("diabetes experiment-2")
mlflow.start_run(experiment_id=exp.experiment_id)

diabetes = datasets.load_diabetes()

if not os.path.exists("data_dir"):
    os.mkdir("data_dir")
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    df["target"] = diabetes.target
    df.to_csv("data_dir/diabetes.csv", index=False)

X = diabetes.data[:, np.newaxis, 2]
y = diabetes.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Linear Regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

if not os.path.exists("model_dir"):
    os.mkdir("model_dir")
    joblib.dump(model, "model_dir/diabetes_model.joblib")


y_pred = model.predict(X_test)

# Plot the data and the regression line
# Print the coefficients (slope and intercept)
print("Slope (Coefficient):", model.coef_[0])
print("Intercept:", model.intercept_)