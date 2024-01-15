import uuid

from ml_scripts.config import mlflow

# script for supporting test_train module


def generate_experiment_name(prefix: str = "test") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def delete_experiment(experiment_name: str) -> None:
    client = mlflow.tracking.MlflowClient()
    experiment_id = client.get_experiment_by_name(experiment_name).experiment_id
    client.delete_experiment(experiment_id=experiment_id)
