from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago

from utils import default_args, DEFAULT_VOLUME


with DAG(
    "1_generate_data",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=days_ago(3),
) as dag:
    start_task = DummyOperator(task_id="begin-generate-data")
    download_data = DockerOperator(
        task_id="docker-airflow-download",
        image="airflow-download",
        command="/data/raw/{{ ds }}",
        network_mode="bridge",
        do_xcom_push=False,
        volumes=[DEFAULT_VOLUME],
    )

    end_task = DummyOperator(task_id="end-generate-data")

    start_task >> download_data >> end_task
