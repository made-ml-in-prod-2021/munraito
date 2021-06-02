import os
import pickle

import pandas as pd
import click
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import mlflow
import logging


@click.command("validate")
@click.option("--data-dir")
@click.option("--transformer-dir")
@click.option("--model-name")
def val(data_dir: str, transformer_dir: str, model_name: str):
    logger = logging.getLogger("evaluate_model")
    # preprocess
    with open(os.path.join(transformer_dir, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    val = pd.read_csv(os.path.join(data_dir, "val.csv"), index_col=0)
    val_X, val_y = val.drop("target", 1), val["target"]
    val_X = scaler.transform(val_X)

    # get the model from MLFLOW
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URL"])
    client = mlflow.tracking.MlflowClient()
    last_model_data = sorted(
        client.search_model_versions(f"name='{model_name}'"),
        key=lambda x: x.last_updated_timestamp,
        reverse=True,
    )[0]
    logger.warning("warn message")
    logger.warning(last_model_data)
    logger.warning(last_model_data.name + ", " + str(last_model_data.version))
    logger.warning(os.getcwd())
    logger.warning(os.listdir(path="."))
    model = mlflow.sklearn.load_model(
        f"models:/{last_model_data.name}/{str(last_model_data.version)}"
    )
    # model = mlflow.sklearn.load_model(f"models:/{model_name}/1")

    # run experiment
    mlflow.set_experiment(f"validate_{model_name}")
    with mlflow.start_run():
        preds = model.predict(val_X)
        mlflow.log_metric("roc_auc_score", roc_auc_score(val_y, preds))
        mlflow.log_metric("accuracy_score", accuracy_score(val_y, preds))
        mlflow.log_metric("f1_score", f1_score(val_y, preds))

    client.transition_model_version_stage(
        name=last_model_data.name,  # model_name,
        version=last_model_data.version,  #  version=1,
        stage="Production",
    )


if __name__ == "__main__":
    val()
