from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from airflow.sensors.filesystem import FileSensor

from utils import default_args, DEFAULT_VOLUME, ARTIFACT_VOLUME, mlflow_env, model_name


with DAG(
    "2_train_model",
    default_args=default_args,
    schedule_interval="@weekly",
    start_date=days_ago(3),
) as dag:
    start_task = DummyOperator(task_id="begin-train-pipeline")
    data_await = FileSensor(
        task_id="await-features",
        poke_interval=10,
        retries=100,
        filepath="data/raw/{{ ds }}/data.csv",
    )
    target_await = FileSensor(
        task_id="await-target",
        poke_interval=10,
        retries=100,
        filepath="data/raw/{{ ds }}/target.csv",
    )
    split = DockerOperator(
        task_id="split-data",
        image="airflow-split",
        command="--input-dir /data/raw/{{ ds }} --output-dir /data/split/{{ ds }}",
        network_mode="bridge",
        do_xcom_push=False,
        volumes=[DEFAULT_VOLUME],
    )
    preprocess = DockerOperator(
        task_id="preprocess-data",
        image="airflow-preprocess",
        command="--input-dir /data/split/{{ ds }} --output-dir /data/processed/{{ ds }}"
        " --transformer-dir /data/transformers/{{ ds }}",
        network_mode="bridge",
        do_xcom_push=False,
        volumes=[DEFAULT_VOLUME],
    )

    train = DockerOperator(
        task_id="train-model",
        image="airflow-train",
        command="--data-dir /data/processed/{{ ds }}" f" --model-name {model_name}",
        network_mode="host",
        private_environment=mlflow_env,
        do_xcom_push=False,
        volumes=[DEFAULT_VOLUME, ARTIFACT_VOLUME],
    )
    validate = DockerOperator(
        task_id="evaluate-model",
        image="airflow-validate",
        command="--data-dir /data/split/{{ ds }} --transformer-dir /data/transformers/{{ ds }}"
        f" --model-name {model_name}",
        network_mode="host",
        private_environment=mlflow_env,
        do_xcom_push=False,
        volumes=[DEFAULT_VOLUME, ARTIFACT_VOLUME],
    )
    end_task = DummyOperator(task_id="end-train-pipeline")

    (
        start_task
        >> [data_await, target_await]
        >> split >> preprocess >> train >> validate
        >> end_task
    )
