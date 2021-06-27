import os
import pickle
import pandas as pd
import numpy as np
import click
from sklearn.preprocessing import StandardScaler


@click.command("preprocess")
@click.option("--input-dir")
@click.option("--output-dir")
@click.option("--transformer-dir")
def preprocess(input_dir: str, output_dir: str, transformer_dir: str):
    train = pd.read_csv(os.path.join(input_dir, "train.csv"), index_col=0)
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train.drop("target", 1))

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(transformer_dir, exist_ok=True)
    train["target"].to_csv(os.path.join(output_dir, "train_target.csv"))
    np.savetxt(
        os.path.join(output_dir, "train_scaled.csv"), scaled_train, delimiter=","
    )
    with open(os.path.join(transformer_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)


if __name__ == "__main__":
    preprocess()
