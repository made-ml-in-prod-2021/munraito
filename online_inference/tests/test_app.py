from fastapi.testclient import TestClient
from app import app
from src.features import COL_ORDER

client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == "Entry point for the inference"


def test_predict_correct_request():
    with client:
        req_data = [55.50565680329502, 0.0, 1.0, 88.43927869835709,
                    219.85520194052953, 1.0, 1.0, 159.39106072419912,
                    0.0, -0.8046441507161775, 2.0, 1.0, 2.0]
        req_feats = COL_ORDER
        response = client.get("/predict/",
                              json={"data": [req_data], "feature_names": req_feats}
                              )
        assert response.status_code == 200
        assert response.json() == [{'disease': 1}]


def test_predict_incorrect_json():
    with client:
        response = client.get("/predict/",
                              json={"incorrect json": 0}
                              )
        assert response.status_code == 422
        # assert response.json() == [f"Input data is not in the right format ({HeartDiseaseModel})"]


def test_predict_empty_data():
    with client:
        response = client.get("/predict/",
                              json={"data": [], "feature_names": COL_ORDER}
                              )
        assert response.status_code == 400
        assert response.json() == {"detail": "Input data list is empty"}


def test_predict_incorrect_amount_of_features():
    with client:
        req_feats = COL_ORDER[:]
        req_feats.append("additional_feature")
        response = client.get("/predict/",
                              json={"data": [[0] * 14], "feature_names": req_feats}
                              )
        assert response.status_code == 422
        assert response.json()["detail"][0]["msg"] == "ensure this value has at most 13 items"


def test_predict_incorrect_col_order():
    with client:
        req_feats = ["sex", "age", "cp", "trestbps", "chol",
                    "fbs", "restecg", "exang", "oldpeak", "thalach",
                    "ca", "slope", "thal"]
        response = client.get("/predict/",
                              json={"data": [[0] * 13], "feature_names": req_feats}
                              )
        assert response.status_code == 400
        assert response.json() == {"detail": "Column order is incorrect"}


def test_predict_incorrect_feature_format():
    with client:
        req_data = [55.50565680329502, 0.67161, 1.0, 88.43927869835709,
                    219.85520194052953, 1.0, 1.0, 159.39106072419912,
                    0.0, -0.8046441507161775, 2.0, 1.0, 2.0]
        response = client.get("/predict/",
                              json={"data": [req_data], "feature_names": COL_ORDER}
                              )
        assert response.status_code == 400
        assert response.json() == {"detail": "'sex' feature is not categorical"}
        req_data = ['aaa', 0.67161, 1.0, 88.43927869835709,
                    219.85520194052953, 1.0, 1.0, 159.39106072419912,
                    0.0, -0.8046441507161775, 2.0, 1.0, 2.0]
        response = client.get("/predict/",
                              json={"data": [req_data], "feature_names": COL_ORDER}
                              )
        assert response.status_code == 422
        assert response.json()["detail"][0]["msg"] == "value is not a valid float"
