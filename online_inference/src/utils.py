import pickle
import yaml
import os
import logging
from typing import List, Union, NoReturn

import pandas as pd
from fastapi import HTTPException
from pydantic import BaseModel, conlist
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer

from src.features import CAT_FEATS, NUM_FEATS, COL_ORDER

SklearnClassifierModel = Union[RandomForestClassifier, LogisticRegression]
PATH_TO_CONFIG = "src/main_config.yaml"


def create_logger(name: str, log_config: dict):
    logger = logging.getLogger(name)
    logger.setLevel(log_config["level"])
    simple_formatter = logging.Formatter(
        fmt=log_config["format"], datefmt=log_config["date_format"]
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_config["level"])
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    return logger


def load_object(path: str):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj


def load_config():
    path = os.getenv("PATH_TO_CONFIG") or PATH_TO_CONFIG
    with open(path) as f:
        config = yaml.safe_load(f)
    return config


class HeartDiseaseModel(BaseModel):
    data: List[conlist(Union[float, int, None], min_items=13, max_items=13)]
    feature_names: List[str]


class ModelResponse(BaseModel):
    disease: int


def validate_data(request: HeartDiseaseModel) -> NoReturn:
    if not isinstance(request, HeartDiseaseModel):
        raise HTTPException(
            status_code=400,
            detail=f"Input data is not in the right format ({HeartDiseaseModel})",
        )
    if not len(request.data):
        raise HTTPException(status_code=400, detail="Input data list is empty")
    redundant_feats = set(request.feature_names) - set(COL_ORDER)
    if len(redundant_feats) > 0:
        raise HTTPException(status_code=400, detail=f"{redundant_feats} are redundant")
    missing_feats = set(COL_ORDER) - set(request.feature_names)
    if len(missing_feats) > 0:
        raise HTTPException(status_code=400, detail=f"{missing_feats} are missing")
    if request.feature_names != COL_ORDER:
        raise HTTPException(status_code=400, detail="Column order is incorrect")
    for feat, val in zip(request.feature_names, request.data[0]):
        if feat in CAT_FEATS and not val.is_integer() and val is not None:
            raise HTTPException(
                status_code=400, detail=f"'{feat}' feature is not categorical"
            )
        if feat in NUM_FEATS and not isinstance(val, float) and val is not None:
            raise HTTPException(status_code=400, detail=f"{feat} is not numerical")


def make_predict(
    data: List,
    feature_names: List[str],
    model: SklearnClassifierModel,
    transfomer: ColumnTransformer,
) -> List[ModelResponse]:
    df = pd.DataFrame(data, columns=feature_names)
    transformed_df = pd.DataFrame(transfomer.transform(df))
    preds = model.predict(transformed_df)
    return [ModelResponse(disease=preds)]
