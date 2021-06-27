import sys
import pytest
from airflow.models import DagBag

sys.path.append("dags")


@pytest.fixture()
def dag_bag():
    return DagBag(dag_folder="dags/", include_examples=False)


def test_dag_bag_correct_imports(dag_bag):
    assert dag_bag.dags is not None
    assert dag_bag.import_errors == {}


def test_generate_data_dag_loaded(dag_bag):
    assert "1_generate_data" in dag_bag.dags
    assert len(dag_bag.dags["1_generate_data"].tasks) == 3


def test_generate_data_dag_structure(dag_bag):
    structure = {
        "begin-generate-data": ["docker-airflow-download"],
        "docker-airflow-download": ["end-generate-data"],
        "end-generate-data": [],
    }
    dag = dag_bag.dags["1_generate_data"]
    for name, task in dag.task_dict.items():
        assert set(structure[name]) == task.downstream_task_ids


def test_train_model_dag_loaded(dag_bag):
    assert "2_train_model" in dag_bag.dags
    assert len(dag_bag.dags["2_train_model"].tasks) == 8


def test_train_model_dag_structure(dag_bag):
    structure = {
        "begin-train-pipeline": ["await-target", "await-features"],
        "await-features": ["split-data"],
        "await-target": ["split-data"],
        "split-data": ["preprocess-data"],
        "preprocess-data": ["train-model"],
        "train-model": ["evaluate-model"],
        "evaluate-model": ["end-train-pipeline"],
        "end-train-pipeline": [],
    }
    dag = dag_bag.dags["2_train_model"]
    for name, task in dag.task_dict.items():
        assert set(structure[name]) == task.downstream_task_ids


def test_predictions_dag_loaded(dag_bag):
    assert "3_make_predicts" in dag_bag.dags
    assert len(dag_bag.dags["3_make_predicts"].tasks) == 6


def test_predictions_dag_structure(dag_bag):
    structure = {
        "begin-inference": ["await-features", "await-scaler", "await-model"],
        "await-features": ["generate-predicts"],
        "await-scaler": ["generate-predicts"],
        "await-model": ["generate-predicts"],
        "generate-predicts": ["end-inference"],
        "end-inference": [],
    }
    dag = dag_bag.dags["3_make_predicts"]
    for name, task in dag.task_dict.items():
        assert set(structure[name]) == task.downstream_task_ids
