import os

from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from airflow.models import Variable
from airflow.sensors.filesystem import FileSensor

from utils import default_args

Variable.set("MODEL_DIR", "/data/models")
with DAG(
    "3_make_predicts",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=days_ago(3)
) as dag:
    start_task = DummyOperator(task_id='begin-inference')
    data_await = FileSensor(
        task_id="await_features",
        poke_interval=10,
        retries=100,
        filepath="data/raw/{{ ds }}/data.csv"
    )
    predict = DockerOperator(
        task_id="generate-predicts",
        image="munraito/airflow-predict",
        command="--input-dir /data/raw/{{ ds }} --output-dir /data/predictions/{{ ds }}"
                " --model-dir {{ var.value.MODEL_DIR }}/{{ ds }}/",
        network_mode="bridge",
        do_xcom_push=False,
        volumes=["/Users/munraito/MADE/ml_in_prod/munraito/airflow_ml_dags/data:/data"]
    )
    end_task = DummyOperator(task_id='end-inference')

    start_task >> data_await >> predict >> end_task
