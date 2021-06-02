import os
import pickle

import pandas as pd
import click
import mlflow


@click.command("predict")
@click.option("--input-dir")
@click.option("--output-dir")
@click.option("--model-name")
@click.option("--transformer-dir")
def predict(input_dir: str, output_dir: str, model_name: str, transformer_dir: str):
    data = pd.read_csv(os.path.join(input_dir, "data.csv"), index_col=0)
    with open(os.path.join(transformer_dir, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URL"])
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/Production")

    data["predict"] = model.predict(scaler.transform(data))

    os.makedirs(output_dir, exist_ok=True)
    data.to_csv(os.path.join(output_dir, "predictions.csv"))


if __name__ == "__main__":
    predict()
