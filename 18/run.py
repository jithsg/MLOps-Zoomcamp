import mlflow

experiment_name = "ElasticNet"
entry_point ="Training"

mlflow.set_tracking_uri("http://localhost:5000")

mlflow.projects.run(uri=".",
                    experiment_name=experiment_name,
                    entry_point=entry_point,
                    env_manager="conda")