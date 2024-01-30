import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path
import sklearn, joblib, cloudpickle
import pandas as pd

if mlflow.active_run():
    # If there is an active run, end it
    mlflow.end_run()

mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

exp = mlflow.set_experiment("my-experiment_tracking-sign")

# Start a new run and specify the run name
mlflow.start_run(run_name="my-run-tracking-2")

print(f'The current experiment is {exp}')
print(f'The current experiment id is {exp.experiment_id}')

db = load_diabetes()
df = pd.DataFrame(data=db.data, columns=db.feature_names)

# Add the target column to the DataFrame
df['target'] = db.target

# Create the "diabetes_dataset" directory if it doesn't exist
data_dir = "diabetes_dataset"
if not Path(data_dir).exists():
    Path(data_dir).mkdir()

# Save the DataFrame as a CSV file in the "diabetes_dataset" directory
csv_file_path = Path(data_dir) / 'diabetes_dataset.csv'
df.to_csv(csv_file_path, index=False)

X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

# Create and train models.
rf = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
rf.fit(X_train, y_train)

# Use the model to make predictions on the test dataset.
predictions = rf.predict(X_test)

sklearn_model_path = "sklearn_model.pkl"
joblib.dump(rf, sklearn_model_path)

# Define the artifacts dictionary
artifacts = {
    "sklearn_model": sklearn_model_path,
    "data": str(csv_file_path)  # Convert the Path object to a string
}

class SklearnWrapper(mlflow.pyfunc.PythonModel):
    
    def load_context(self, context):
        self.sklearn_model = joblib.load(context.artifacts["sklearn_model"])

    def predict(self, model_input):
        print(f'The model input is {model_input}')
        return self.sklearn_model.predict(pd.DataFrame(model_input))

conda_env ={
    "channels": [
        "defaults"
    ],
    "dependencies": [
        "python={}".format(3.10),
        "pip",
        {
            "pip": [
                "mlflow=={}".format(mlflow.__version__),
                "cloudpickle=={}".format(cloudpickle.__version__),
                "scikit-learn=={}".format(sklearn.__version__)
            ]
        }
    ],
    "name": "sklearn_env"
}

mlflow.pyfunc.log_model(
    artifact_path="sklearn_mlflow_pyfunc",
    python_model=SklearnWrapper(),
    artifacts=artifacts,  # Pass the artifacts dictionary here
    code_path=["main.py"],
    conda_env=conda_env
)

ld = mlflow.pyfunc.load_model(model_uri='runs:/8a3d42d41dff4f0c9e57ae11a68e81bf/sklearn_mlflow_pyfunc')

predicteds = ld.predict(X_test)

r2_score = r2_score(y_test, predicteds)

print(f'The R2 score is {r2_score}')
mlflow.end_run()
