import mlflow
import joblib
import pandas as pd
from mlflow import MlflowClient
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import numpy as np
from mlflow.entities import ViewType

mlflow.set_tracking_uri("http://127.0.0.1:5000")

client= MlflowClient()


# run= client.get_run('183c9aaaaa29402d94511241f85e459b')

# metrics = client.get_metric_history(run.info.run_id, "rmse")

# for metric in metrics:
#     print(f"Step: {metric.step}, Value: {metric.value}")
# client.set_terminated(run.info.run_id, status="FINISHED")

# client.create_registered_model(
#     name="model_1",
#     tags= {
#         'framework': 'sklearn',
#         'version': '0.0.1',
#         'type': 'regression'
#     },
#     description="model_1"
# )

# client.create_model_version(
#     name="model_1",
#     source= "runs:/80c84537301747c1aa816e7045425b40/model",
    
# )
# client.transition_model_version_stage(
#     name="model_1",
#     stage="Production",
#     version='1',
#     archive_existing_versions=True
# )

mv=client.get_model_version(
    name="model_1",
    version="1"
)
print('Name:', mv.name)
print('Current stage:', mv.current_stage)