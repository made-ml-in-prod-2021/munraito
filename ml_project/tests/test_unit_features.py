import os
import pandas as pd
import numpy as np

from src.scaler import CustomStandardScaler
from src.build_features import build_transformer, make_features, serialize_transformer, deserialize_transformer
from .constants import *


def test_custom_scaler(fake_df):
    num_df = pd.read_csv(fake_df)[NUM_FEATS]
    for feat in NUM_FEATS:
        curr = num_df[feat]
        expected_col = (curr - curr.mean()) / curr.std()
        scaler = CustomStandardScaler()
        scaler.fit(curr)
        transformed_col = scaler.transform(curr)
        assert np.allclose(expected_col, transformed_col)


def test_transformer_pipelines(fake_df, train_params):
    fake_df = pd.read_csv(fake_df)
    transformer = build_transformer(train_params)
    transformer.fit(fake_df)
    transformed_df = make_features(transformer, fake_df)
    expected_rows = fake_df.shape[0]
    expected_cols = 30
    assert not pd.isnull(transformed_df).any().any()
    assert expected_rows, expected_cols == transformed_df.shape


def test_save_and_load_transfomer(fake_df, train_params):
    fake_df = pd.read_csv(fake_df)
    transformer = build_transformer(train_params)
    transformer.fit(fake_df)
    serialize_transformer(transformer, train_params.output_transformer_path)
    assert os.path.exists(train_params.output_transformer_path)
    transformed_df = make_features(transformer, fake_df)
    loaded_transformer = deserialize_transformer(train_params.output_transformer_path)
    loaded_transformer_df = make_features(loaded_transformer, fake_df)
    assert not pd.isnull(loaded_transformer_df).any().any()
    assert loaded_transformer_df.equals(transformed_df)
