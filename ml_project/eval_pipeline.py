import os
import logging
import pandas as pd
from typing import NoReturn
from omegaconf import DictConfig
import hydra

from src.configs.eval_params import EvalPipelineParamsSchema, EvalPipelineParams
from src.build_features import make_features, deserialize_transformer
from src.models import predict_model, deserialize_model

logger = logging.getLogger("eval_pipeline")
formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
logger.setLevel(logging.INFO)


def eval_pipeline(params: EvalPipelineParams) -> NoReturn:
    logger.info(f"start eval pipeline with params {params}")
    df = pd.read_csv(params.input_df_path)
    logger.info(f"df.shape is {df.shape}")
    transformer = deserialize_transformer(params.transformer_path)
    transformed_df = make_features(transformer, df)
    logger.info(f"transformed_df.shape is {transformed_df.shape}")

    model = deserialize_model(params.model_path)
    preds = predict_model(
        model,
        transformed_df,
    )
    logger.info(f"preds.shape is {preds.shape}")
    # preds_df = pd.DataFrame(preds)
    pd.DataFrame(preds).to_csv(params.preds_path, header=False)
    logger.info(f"predicts saved to the {params.preds_path}")


@hydra.main(config_path="configs", config_name="eval_config")
def eval_pipeline_command(cfg: DictConfig):
    os.chdir(hydra.utils.to_absolute_path("."))
    schema = EvalPipelineParamsSchema()
    params = schema.load(cfg)
    eval_pipeline(params)


if __name__ == "__main__":
    eval_pipeline_command()
