import pickle
from typing import Dict, Union, NoReturn

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from src.configs.train_params import LRParams, RFParams

SklearnClassifierModel = Union[RandomForestClassifier, LogisticRegression]


def train_model(
    features: pd.DataFrame,
    target: pd.Series,
    params: Union[LRParams, RFParams],
) -> SklearnClassifierModel:
    if params.model_type == "RandomForestClassifier":
        model = RandomForestClassifier(
            n_estimators=params.n_estimators,
            criterion=params.criterion,
            random_state=params.random_state,
        )
    elif params.model_type == "LogisticRegression":
        model = LogisticRegression(
            penalty=params.penalty,
            tol=params.tol,
            C=params.C,
            random_state=params.random_state,
        )
    else:
        raise NotImplementedError()
    model.fit(features, target)
    return model


def predict_model(model: SklearnClassifierModel, features: pd.DataFrame) -> np.ndarray:
    predicts = model.predict(features)
    return predicts


def evaluate_model(predicts: np.ndarray, target: pd.Series) -> Dict[str, float]:
    return {
        "roc_auc_score": roc_auc_score(target, predicts),
        "accuracy_score": accuracy_score(target, predicts),
        "f1_score": f1_score(target, predicts),
    }


def serialize_model(model: SklearnClassifierModel, output: str) -> NoReturn:
    with open(output, "wb") as f:
        pickle.dump(model, f)


def deserialize_model(input_: str) -> SklearnClassifierModel:
    with open(input_, "rb") as f:
        model = pickle.load(f)
    return model
