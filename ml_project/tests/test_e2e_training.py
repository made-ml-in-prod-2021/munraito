import os

from train_pipeline import train_pipeline
from src.configs.train_params import TrainingPipelineParams


def test_train_pipeline(tmpdir, train_params: TrainingPipelineParams):
    metrics = train_pipeline(train_params)
    assert os.path.exists(tmpdir.join("metric.json"))
    assert os.path.exists(tmpdir.join("model.pkl"))
    assert os.path.exists(tmpdir.join("transformer.pkl"))
    assert metrics["roc_auc_score"] > 0.6
    assert metrics["accuracy_score"] > 0.6
    assert metrics["f1_score"] > 0.6
