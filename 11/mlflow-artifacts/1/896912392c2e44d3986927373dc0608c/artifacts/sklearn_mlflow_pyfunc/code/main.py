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

exp = mlflow.set_experiment("diabetes experiment-1")

mlflow.start_run(experiment_id=exp.experiment_id)

# Load the diabetes dataset
diabetes = datasets.load_diabetes()

if not os.path.exists("data_dir"):
    os.mkdir("data_dir")
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    df["target"] = diabetes.target
    df.to_csv("data_dir/diabetes.csv", index=False)

# Use only one feature for simplicity (e.g., feature 2)
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


artifacts = {"model": "model_dir/diabetes_model.joblib", "data": "data_dir/diabetes.csv"}

# Make predictions on the test data
y_pred = model.predict(X_test)

# Plot the data and the regression line
# Print the coefficients (slope and intercept)
print("Slope (Coefficient):", model.coef_[0])
print("Intercept:", model.intercept_)

class SklearnWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self,  context):
        self.model = joblib.load(context.artifacts["model"])    
        

    def predict(self, context, model_input):
        return self.model.predict(model_input)


conda_env = {
    "channels": ["defaults"],
    "dependencies": [
        "python={}".format(3.10),
        "pip",
        {
            "pip": [
                "mlflow=={}".format(mlflow.__version__),
                "scikit-learn=={}".format(sklearn.__version__),
                "cloudpickle=={}".format(cloudpickle.__version__),
            ],
        },
    ],
    "name": "sklearn_env",
}

mlflow.pyfunc.log_model(
    artifact_path="sklearn_mlflow_pyfunc",
    python_model=SklearnWrapper(),
    artifacts=artifacts,
    code_path=["main.py"],
    conda_env=conda_env
)


# ld = mlflow.pyfunc.load_model(model_uri="runs:/ee0c9144a0e941168dcdebe82e9cae47/sklear_mlflow_pyfunc")
# predicted_qualities=ld.predict(X_test)
# print(predicted_qualities)

mlflow.end_run()