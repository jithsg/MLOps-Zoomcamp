import mlflow
import joblib
mlflow.set_tracking_uri("http://localhost:5000")

# Start an MLflow experiment
experiment_name = "Iris RandomForest Experiment"
mlflow.set_experiment(experiment_name)


loaded_model = joblib.load(open('random_forest_iris_model.pkl', 'rb'))
with mlflow.start_run():
    # Log parameters (hyperparameters of your model)
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)

    # Log model
    mlflow.sklearn.log_model(loaded_model, "model", registered_model_name="Iris RandomForest Model")

    # Optionally, log other metrics


print("Model logged to MLflow.")