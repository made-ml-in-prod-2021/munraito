import os
import pickle

import pandas as pd
import click


@click.command("predict")
@click.option("--input-dir")
@click.option("--output-dir")
@click.option("--model-dir")
def predict(input_dir: str, output_dir: str, model_dir: str):
    data = pd.read_csv(os.path.join(input_dir, "data.csv"), index_col=0)
    with open(os.path.join(model_dir, "logreg.pkl"), "rb") as m,\
            open(os.path.join(model_dir, "scaler.pkl"), "rb") as s:
        model = pickle.load(m)
        scaler = pickle.load(s)
    data["predict"] = model.predict(scaler.transform(data))

    os.makedirs(output_dir, exist_ok=True)
    data.to_csv(os.path.join(output_dir, "predictions.csv"))


if __name__ == '__main__':
    predict()