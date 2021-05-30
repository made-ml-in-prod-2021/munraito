from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from airflow.sensors.filesystem import FileSensor

from utils import default_args


with DAG(
    "2_train_model",
    default_args=default_args,
    schedule_interval="@weekly",
    start_date=days_ago(3)
) as dag:
    start_task = DummyOperator(task_id='begin-train-pipeline')
    data_await = FileSensor(
        task_id="await_features",
        poke_interval=10,
        retries=100,
        filepath="data/raw/{{ ds }}/data.csv"
    )
    target_await = FileSensor(
        task_id="await_target",
        poke_interval=10,
        retries=100,
        filepath="data/raw/{{ ds }}/target.csv"
    )
    split = DockerOperator(
        task_id="split-data",
        image="munraito/airflow-split",
        command="--input-dir /data/raw/{{ ds }} --output-dir /data/split/{{ ds }}",
        network_mode="bridge",
        do_xcom_push=False,
        volumes=["/Users/munraito/MADE/ml_in_prod/munraito/airflow_ml_dags/data:/data"]
    )
    preprocess = DockerOperator(
        task_id="preprocess-data",
        image="munraito/airflow-preprocess",
        command="--input-dir /data/split/{{ ds }} --output-dir /data/processed/{{ ds }}"
                " --model-dir /data/models/{{ ds }}",
        network_mode="bridge",
        do_xcom_push=False,
        volumes=["/Users/munraito/MADE/ml_in_prod/munraito/airflow_ml_dags/data:/data"]
    )
    train = DockerOperator(
        task_id="train-model",
        image="munraito/airflow-train",
        command="--data-dir /data/processed/{{ ds }} --model-dir /data/models/{{ ds }}",
        network_mode="bridge",
        do_xcom_push=False,
        volumes=["/Users/munraito/MADE/ml_in_prod/munraito/airflow_ml_dags/data:/data"]
    )
    validate = DockerOperator(
        task_id="evaluate-model",
        image="munraito/airflow-validate",
        command="--data-dir /data/split/{{ ds }} --model-dir /data/models/{{ ds }}",
        network_mode="bridge",
        do_xcom_push=False,
        volumes=["/Users/munraito/MADE/ml_in_prod/munraito/airflow_ml_dags/data:/data"]
    )
    end_task = DummyOperator(task_id='end-train-pipeline')

    start_task >> [data_await, target_await] >> split >> preprocess >> train >> validate >> end_task
