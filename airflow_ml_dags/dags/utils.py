from datetime import timedelta

default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5)
}

DEFAULT_VOLUME = "/Users/munraito/MADE/ml_in_prod/munraito/airflow_ml_dags/data:/data"
