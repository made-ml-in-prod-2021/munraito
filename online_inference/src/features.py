from dataclasses import dataclass


@dataclass()
class NumFeat:
    name: str
    mean: float
    std: float


@dataclass()
class CatFeat:
    name: str
    nunique: int


CAT_FEATS = {
    "sex": CatFeat("sex", 2),
    "cp": CatFeat("cp", 4),
    "fbs": CatFeat("fbs", 2),
    "restecg": CatFeat("restecg", 3),
    "exang": CatFeat("exang", 2),
    "slope": CatFeat("slope", 3),
    "ca": CatFeat("ca", 5),
    "thal": CatFeat("thal", 4),
}
NUM_FEATS = {
    "age": NumFeat("age", 54.37, 9.08),
    "trestbps": NumFeat("trestbps", 131.62, 17.54),
    "chol": NumFeat("chol", 246.26, 51.83),
    "thalach": NumFeat("thalach", 149.65, 22.91),
    "oldpeak": NumFeat("oldpeak", 1.04, 1.16),
}

COL_ORDER = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]
