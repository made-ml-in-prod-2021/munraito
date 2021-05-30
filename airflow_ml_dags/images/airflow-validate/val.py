import os
import pickle
import json

import pandas as pd
import click
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score


@click.command("validate")
@click.option("--data-dir")
@click.option("--model-dir")
def val(data_dir: str, model_dir: str):
    with open(os.path.join(model_dir, "logreg.pkl"), "rb") as m,\
            open(os.path.join(model_dir, "scaler.pkl"), "rb") as s:
        model = pickle.load(m)
        scaler = pickle.load(s)
    val = pd.read_csv(os.path.join(data_dir, "val.csv"), index_col=0)

    val_X, val_y = val.drop('target', 1), val['target']
    val_X = scaler.transform(val_X)
    preds = model.predict(val_X)
    metrics = {
        "roc_auc_score": roc_auc_score(val_y, preds),
        "accuracy_score": accuracy_score(val_y, preds),
        "f1_score": f1_score(val_y, preds),
    }

    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f)


if __name__ == '__main__':
    val()
