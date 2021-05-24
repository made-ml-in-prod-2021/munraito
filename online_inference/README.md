online inference (HW 2)
=============================
Code for REST app which can predict heart diseases.
Dockerized. Built on FastAPI + uvicorn

Project Structure
------------

    ├── model                      <- pickled artifacts folder
    │   └── model.pkl              <- pre-trained model
    │   └── transformer.pkl        <- pre-trained transformer
    ├── src                        <- all useful code for app
    │   └── features.py            <- constants for all features of the model
    │   └── scaler.py              <- custom Scaler class (copied from ml_project)
    │   └── utils.py               <- different functions which supports the main solution files
    │   └── main_config.yaml       <- some configs
    ├── tests                      <- tests folder
    │   └── test_app.py            <- testing /predict/ API
    ├── app.py                     <- main entry point (FastAPI application)
    ├── Dockerfile                 <- commands to create docker image
    ├── generate_requests.py       <- generate fake data and fire it to the /predict/
    ├── requiriments.txt           <- all necessary dependencies for docker image
    ├── README.md                  <- This very document.

--------
#### Using ready-made docker image
```
docker pull munraito/online_inference:v3
docker run -p 8000:8000 munraito/online_inference:v3
```
#### Building from scratch
```
docker build -t munraito/online_inference:v3 .
docker push munraito/online_inference:v3
```
#### Fire requests and run pytest (locally)
```
python generate_requests.py
pytest -v --cov
```
#### What was made to optimize docker image
Initial image size: **1.33G**, building time is **~2 minutes**.

Then,
 - the base image was changed (earlier python, slim version: **python:3.6-slim-stretch**)
 - only necessary dependencies in requirements.txt were used
 - number of layers were minimized
 
 Final image size on docker hub: **192M** (compressed), building time is **~50 seconds**.
