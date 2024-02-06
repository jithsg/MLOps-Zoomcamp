import mlflow
import mlflow.sklearn
import numpy as np
from data import X_train, X_val, y_train, y_val
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import ParameterGrid
from params import elasticnet_param_grid
from utils import eval_metrics

# Enable automatic logging for scikit-learn models.
# mlflow.sklearn.autolog()

# Loop through the hyperparameter combinations and log results in separate runs
for params in ParameterGrid(elasticnet_param_grid):
    with mlflow.start_run():
        lr = ElasticNet(**params)
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_val)

        # Calculate and log evaluation metrics
        metrics = eval_metrics(y_val, y_pred)
        
        # Log each metric manually (optional, as autolog captures most metrics)
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)
        
        # Log parameters manually (optional, autolog captures this as well)
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)

        # Log the model manually (optional, autolog does this automatically)
        # mlflow.sklearn.log_model(lr, "model")

        # Logging the inputs such as dataset
        # mlflow.log_input(
        #     mlflow.data.from_numpy(X_train.toarray()),
        #     context='Training dataset'
        # )

        # mlflow.log_input(
        #     mlflow.data.from_numpy(X_val.toarray()),
        #     context='Validation dataset'
        # )

        # Logging hyperparameters
    