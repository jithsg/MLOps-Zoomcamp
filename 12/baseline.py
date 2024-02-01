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
from sklearn.dummy import DummyRegressor
from mlflow.models import MetricThreshold


mlflow.set_tracking_uri("http://localhost:5000")    

mlflow.sklearn.autolog()

exp = mlflow.set_experiment("diabetes experiment-2")
mlflow.start_run(experiment_id=exp.experiment_id)

diabetes = datasets.load_diabetes()

df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df["target"] = diabetes.target
if not os.path.exists("data_dir"):
    os.mkdir("data_dir")
    df.to_csv("data_dir/diabetes.csv", index=False)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Linear Regression model
model = LinearRegression()
baseline_model = DummyRegressor(strategy="mean")

# Fit the model to the training data
model.fit(X_train, y_train)
baseline_model.fit(X_train, y_train)


if not os.path.exists("model_dir"):
    os.mkdir("model_dir")
    joblib.dump(model, "model_dir/diabetes_model.joblib")


y_pred = model.predict(X_test)

# Plot the data and the regression line
# Print the coefficients (slope and intercept)
print("Slope (Coefficient):", model.coef_[0])
print("Intercept:", model.intercept_)

artifacts_uri = mlflow.get_artifact_uri("model")

mlflow.sklearn.log_model(baseline_model, "baseline_model")
baseline_model_uri = mlflow.get_artifact_uri("baseline_model")

X_test['target'] = y_test


# Retrieve the active run ID
run_id = mlflow.active_run().info.run_id

# Construct the model URI
model_uri = f"runs:/{run_id}/model"
baseline_model_uri = f"runs:/{run_id}/baseline_model"

thresholds = {
    "mean_squared_error": MetricThreshold(
        threshold=0.5,  # accuracy should be >=0.5
        min_absolute_change=0.05,  # accuracy should be at least 0.05 greater than baseline model accuracy
        min_relative_change=0.05,  # accuracy should be at least 5 percent greater than baseline model accuracy
        greater_is_better=False
    ),
}


mlflow.evaluate(
    model_uri,
    X_test,
    targets='target',  # Use the actual target values from your test set
    model_type="regressor",
    evaluators=["default"],
    validation_thresholds=thresholds,
    baseline_model=baseline_model_uri
)
print(X_test)
mlflow.end_run()


