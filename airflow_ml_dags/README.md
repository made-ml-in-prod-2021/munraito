# Airflow & MLflow
Full pipeline of training & predicting daily generated batches
 of data using Airflow and MLflow. Dockerized.
### Build solution
```
GUSER=<email> GPASS=<pass> docker compose up --build
```
### Correct stopping
```
^C (cntrl-C)
docker compose down
```
### Run tests
```
pytest -v
```