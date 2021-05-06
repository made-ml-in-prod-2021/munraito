import logging
import json
import os

from omegaconf import DictConfig
import hydra
import pandas as pd
from sklearn.model_selection import train_test_split

from src.configs.train_params import (
    TrainingPipelineParamsSchema,
    TrainingPipelineParams,
)
from src.build_features import build_transformer, make_features, serialize_transformer
from src.models import train_model, predict_model, evaluate_model, serialize_model

logger = logging.getLogger("train_pipeline")
logger.setLevel(logging.INFO)


def train_pipeline(params: TrainingPipelineParams):
    logger.info(f"start train pipeline with params {params}")
    df = pd.read_csv(params.input_df_path)
    logger.info(f"df.shape is {df.shape}")
    train, val = train_test_split(
        df, test_size=params.val_size, random_state=params.random_state
    )
    logger.info(f"train.shape is {train.shape}")
    logger.info(f"val.shape is {val.shape}")

    train_feats = train.drop(params.target_col, 1)
    train_target = train[params.target_col]
    transformer = build_transformer(params)
    transformer.fit(train_feats)
    serialize_transformer(transformer, params.output_transformer_path)
    train_feats = make_features(transformer, train_feats)
    logger.info(f"train_feats.shape is {train_feats.shape}")

    model = train_model(train_feats, train_target, params.model)
    val_feats = make_features(transformer, val)
    val_target = val[params.target_col]
    logger.info(f"val_feats.shape is {val_feats.shape}")
    predicts = predict_model(
        model,
        val_feats,
    )

    metrics = evaluate_model(predicts, val_target)
    with open(params.metric_path, "w") as metric_file:
        json.dump(metrics, metric_file)
    logger.info(f"metrics are {metrics}")
    serialize_model(model, params.output_model_path)
    return metrics


@hydra.main(config_path="configs", config_name="train_config")
def train_pipeline_command(cfg: DictConfig):
    os.chdir(hydra.utils.to_absolute_path("."))
    schema = TrainingPipelineParamsSchema()
    params = schema.load(cfg)
    train_pipeline(params)


if __name__ == "__main__":
    train_pipeline_command()
