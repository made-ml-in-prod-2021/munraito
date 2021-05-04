import os
import pytest

from .generate_data import generate_data
from train_pipeline import train_pipeline
from eval_pipeline import eval_pipeline
from src.configs.train_params import LRParams, TrainingPipelineParams
from src.configs.eval_params import EvalPipelineParams


TRAIN_ROWS = 500
EVAL_ROWS = 200
DF_PATH = "data/raw/heart.csv"
CAT_FEATS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
NUM_FEATS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
TARGET_COL = "target"


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


def test_train_pipeline(tmpdir, train_params: TrainingPipelineParams):
    metrics = train_pipeline(train_params)
    assert os.path.exists(tmpdir.join("metric.json"))
    assert os.path.exists(tmpdir.join("model.pkl"))
    assert os.path.exists(tmpdir.join("transformer.pkl"))
    assert metrics["roc_auc_score"] > 0.6
    assert metrics["accuracy_score"] > 0.6
    assert metrics["f1_score"] > 0.6


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
        model_path=tmpdir.join("model.pkl")
    )


def test_eval_pipeline(eval_params: EvalPipelineParams, train_on_fake_df):
    preds = eval_pipeline(eval_params)
    assert preds.shape[0] == EVAL_ROWS
    assert os.path.exists(eval_params.preds_path)
    assert set(preds) == {0, 1}
