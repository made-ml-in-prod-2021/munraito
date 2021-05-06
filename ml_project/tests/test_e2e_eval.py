import pytest
import os

from src.configs.eval_params import EvalPipelineParams
from .generate_data import generate_data
from .constants import EVAL_ROWS, DF_PATH, CAT_FEATS, NUM_FEATS
from eval_pipeline import eval_pipeline
from train_pipeline import train_pipeline


@pytest.fixture
def fake_eval_df(tmpdir):
    df_fio = tmpdir.join("fake_eval_df.csv")
    fake_df = generate_data(EVAL_ROWS, DF_PATH, CAT_FEATS, NUM_FEATS)
    fake_df.to_csv(df_fio, index=False)
    return df_fio


@pytest.fixture
def train_on_fake_df(tmpdir, train_params):
    train_pipeline(train_params)


@pytest.fixture()
def eval_params(tmpdir, fake_eval_df):
    return EvalPipelineParams(
        input_df_path=fake_eval_df,
        preds_path=tmpdir.join("preds.csv"),
        transformer_path=tmpdir.join("transformer.pkl"),
        model_path=tmpdir.join("model.pkl"),
    )


def test_eval_pipeline(eval_params: EvalPipelineParams, train_on_fake_df):
    preds = eval_pipeline(eval_params)
    assert preds.shape[0] == EVAL_ROWS
    assert os.path.exists(eval_params.preds_path)
    assert set(preds) == {0, 1}
