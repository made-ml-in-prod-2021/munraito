import numpy as np
import pandas as pd


def generate_data(
    num_rows: int, df_path, cat_feats, num_feats, target_col=None
) -> pd.DataFrame:
    column_order = ["age", "sex", "cp", "trestbps", "chol",
                    "fbs", "restecg", "thalach", "exang", "oldpeak",
                    "slope", "ca", "thal"]
    float_values = {}
    df = pd.read_csv(df_path)
    for feat in num_feats:
        mu, sigma = df[feat].mean(), df[feat].std()
        float_values[feat] = np.random.normal(mu, sigma, num_rows)
    categorical_values = {}
    for feat in cat_feats:
        categorical_values[feat] = np.random.randint(
            0, df[feat].nunique(), num_rows
        )
    generated = {**float_values, **categorical_values}
    gen_df = pd.DataFrame(generated)[column_order]
    if target_col is not None:
        # np.random.choice([0, 1], num_rows)
        gen_df[target_col] = ((gen_df["cp"] > 1) | (gen_df["oldpeak"] == 0)).astype(
            np.int8
        )
    return gen_df
