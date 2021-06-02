from datetime import timedelta
import os

default_args = {
    "owner": "airflow",
    "email_on_failure": True,
    "email": ["airflow@example.com"],  # , "cvetvitaly@yandex.ru"
    "retries": 1,
    "retry_delay": timedelta(minutes=5)
}
mlflow_env = {
        "MLFLOW_TRACKING_URL": os.environ["MLFLOW_TRACKING_URL"]
}
model_name = os.environ["MODEL_NAME"]

DEFAULT_VOLUME = "/Users/munraito/MADE/ml_in_prod/munraito/airflow_ml_dags/data:/data"
ARTIFACT_VOLUME = "mlrun_data:/mlruns"
