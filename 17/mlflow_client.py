import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("http://localhost:5000")

client = MlflowClient()

# experiment_id = client.create_experiment(name= "Experiment",
#                                         tags={"project":"Project 1"})

# print("Experiment id", experiment_id)

client.delete_experiment("6")