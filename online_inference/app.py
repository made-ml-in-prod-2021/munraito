import os
import time
from typing import List, Optional
import uvicorn
from fastapi import FastAPI
from sklearn.compose import ColumnTransformer

from src.utils import (
    SklearnClassifierModel,
    load_object,
    load_config,
    HeartDiseaseModel,
    ModelResponse,
    validate_data,
    make_predict,
    create_logger,
)


model: Optional[SklearnClassifierModel] = None
transformer: Optional[ColumnTransformer] = None
startup_time: Optional[float] = None
app = FastAPI()
config = load_config()
logger = create_logger("main_app", config["logging"])


@app.get("/")
def main():
    return "Entry point for the inference"


@app.on_event("startup")
def load_model():
    time.sleep(25)
    global model, transformer, startup_time
    model_path = config["model_path"]
    transformer_path = config["transformer_path"]
    if model_path is None or transformer_path is None:
        err = (
            f"PATH_TO_MODEL is {model_path}, PATH_TO_TRANSFORMER is {transformer_path}"
        )
        logger.error(err)
        raise RuntimeError(err)
    model = load_object(model_path)
    transformer = load_object(transformer_path)
    logger.info(f"{model_path} and {transformer_path} successfully loaded")
    startup_time = time.time()


@app.get("/health")
def check_app() -> bool:
    if (time.time() - startup_time) > 120:
        raise RuntimeError("Had to kill you, sorry")
    return not (model is None)


@app.get("/predict/", response_model=List[ModelResponse])
def predict(request: HeartDiseaseModel):
    validate_data(request)
    logger.info("data validation successful")
    return make_predict(request.data, request.feature_names, model, transformer)


if __name__ == "__main__":
    uvicorn.run(
        app, host="0.0.0.0", port=os.getenv("PORT", 8000)
    )
