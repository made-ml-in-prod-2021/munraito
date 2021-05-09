import numpy as np
import pandas as pd
import requests

from src.features import CAT_FEATS, NUM_FEATS, COL_ORDER
from src.utils import create_logger, load_config

NUM_ROWS = 30


def generate_data(
    num_rows: int
) -> pd.DataFrame:
    float_values = {}
    for feat, stat in NUM_FEATS.items():
        mu, sigma = stat.mean, stat.std
        float_values[feat] = np.random.normal(mu, sigma, num_rows)
    categorical_values = {}
    for feat, stat in CAT_FEATS.items():
        categorical_values[feat] = np.random.randint(
            0, stat.nunique, num_rows
        )
    generated = {**float_values, **categorical_values}
    return pd.DataFrame(generated)[COL_ORDER]


if __name__ == "__main__":
    config = load_config()
    logger = create_logger("request_generator", config["logging"])
    fake_df = generate_data(NUM_ROWS)
    request_feats = list(fake_df.columns)
    for i in range(NUM_ROWS):
        request_data = [
            x for x in fake_df.iloc[i].tolist()
        ]
        logger.info(f"request_data: {request_data}")
        response = requests.get(
            f"http://{config['host']}:{config['port']}/predict/",
            json={"data": [request_data], "feature_names": request_feats},
        )
        logger.info(f"status_code: {response.status_code}")
        logger.info(f"response.json:: {response.json()}")
