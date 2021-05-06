import os
import pandas as pd
from pandas_profiling import ProfileReport
from omegaconf import DictConfig
import hydra

from src.configs.train_params import (
    TrainingPipelineParamsSchema,
    TrainingPipelineParams,
)


def build_report(params: TrainingPipelineParams):
    df = pd.read_csv(params.input_df_path)
    ProfileReport(df, title="Pandas Profiling Report").to_file(params.report_path)


@hydra.main(config_path="configs", config_name="train_config")
def build_report_command(cfg: DictConfig):
    os.chdir(hydra.utils.to_absolute_path("."))
    schema = TrainingPipelineParamsSchema()
    params = schema.load(cfg)
    build_report(params)


if __name__ == "__main__":
    build_report_command()
