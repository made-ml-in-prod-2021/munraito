import os
import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from typing import Tuple

from src.models import (
    train_model,
    predict_model,
    evaluate_model,
    serialize_model,
    deserialize_model,
)
from src.build_features import build_transformer, make_features


@pytest.fixture
def preprocess_data(train_params, fake_df) -> Tuple[pd.Series, pd.DataFrame]:
    df = pd.read_csv(fake_df)
    transformer = build_transformer(train_params)
    transformer.fit(df)
    transformed_features = make_features(transformer, df)
    target = df[train_params.target_col]
    return target, transformed_features


def test_train_predict_evaluate_model(train_params, preprocess_data):
    target, transformed_features = preprocess_data
    model = train_model(transformed_features, target, train_params.model)
    assert isinstance(model, LogisticRegression)
    assert target.shape == model.predict(transformed_features).shape
    preds = predict_model(model, transformed_features)
    assert target.shape == preds.shape
    assert {0, 1} == set(preds)
    metrics = evaluate_model(preds, target)
    assert metrics["roc_auc_score"] > 0.7
    assert metrics["accuracy_score"] > 0.7
    assert metrics["f1_score"] > 0.7


def test_save_and_load_model(train_params, preprocess_data):
    target, transformed_features = preprocess_data
    model = train_model(transformed_features, target, train_params.model)
    preds = model.predict(transformed_features)
    serialize_model(model, train_params.output_model_path)
    assert os.path.exists(train_params.output_model_path)
    model = deserialize_model(train_params.output_model_path)
    preds_loaded_model = model.predict(transformed_features)
    assert isinstance(model, LogisticRegression)
    assert np.allclose(preds, preds_loaded_model)
