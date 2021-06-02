import os

import pandas as pd
import numpy as np
import click
import mlflow
from sklearn.linear_model import LogisticRegression


@click.command("train")
@click.option("--data-dir")
@click.option("--model-name")
def train(data_dir: str, model_name: str):
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URL"])
    mlflow.set_experiment(f"train_{model_name}")
    mlflow.sklearn.autolog()
    with mlflow.start_run():
        X = np.loadtxt(os.path.join(data_dir, "train_scaled.csv"), delimiter=",")
        y = pd.read_csv(os.path.join(data_dir, "train_target.csv"), index_col=0)
        model = LogisticRegression()
        model.fit(X, y)
        mlflow.sklearn.log_model(
            sk_model=model, artifact_path="models", registered_model_name=model_name
        )


if __name__ == "__main__":
    train()
