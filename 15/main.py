import os
import warnings
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import mlflow
import mlflow.sklearn
import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--l1_ratio", type=float, default=0.4)
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file (ensure this file exists in your project directory)
    data = pd.read_csv("red-wine-quality.csv")

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    # Set the MLflow tracking URI (make sure the MLflow tracking server is running at this URI)
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    # Set or create the experiment
    mlflow.set_experiment(experiment_name="Project exp")

    # Automatically creates a new run for the experiment
    with mlflow.start_run():

        lr = ElasticNet(alpha=args.alpha, l1_ratio=args.l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("Elasticnet model (alpha={}, l1_ratio={}):".format(args.alpha, args.l1_ratio))
        print("  RMSE: {}".format(rmse))
        print("  MAE: {}".format(mae))
        print("  R2: {}".format(r2))

        # Log model parameters, metrics, and the model itself
        mlflow.log_params({"alpha": args.alpha, "l1_ratio": args.l1_ratio})
        mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})
        mlflow.sklearn.log_model(lr, "model")

if __name__ == "__main__":
    main()
