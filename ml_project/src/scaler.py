from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class TransformException(Exception):
    pass


class CustomStandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, x: np.ndarray) -> "CustomStandardScaler":
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise TransformException('must be fitted first')
        else:
            x = (x - self.mean) / self.std
            return x
