import pytest

from .generate_data import generate_data
from src.configs.train_params import LRParams, TrainingPipelineParams
from src.configs.eval_params import EvalPipelineParams
from train_pipeline import train_pipeline
from .constants import *


@pytest.fixture
def fake_df(tmpdir):
    df_fio = tmpdir.join("fake_train_df.csv")
    fake_df = generate_data(TRAIN_ROWS, DF_PATH, CAT_FEATS, NUM_FEATS, TARGET_COL)
    fake_df.to_csv(df_fio, index=False)
    return df_fio


@pytest.fixture()
def train_params(tmpdir, fake_df):
    return TrainingPipelineParams(
        input_df_path=fake_df,
        output_model_path=tmpdir.join("model.pkl"),
        output_transformer_path=tmpdir.join("transformer.pkl"),
        metric_path=tmpdir.join("metric.json"),
        report_path="",
        categorical_features=CAT_FEATS,
        numerical_features=NUM_FEATS,
        target_col=TARGET_COL,
        val_size=0.2,
        random_state=69,
        model=LRParams(
            model_type="LogisticRegression",
            penalty="l2",
            tol=1e-4,
            C=1.0,
            random_state=69
        )
    )
