ml_project
==============================

Production ready ML-project

Author: Tsvetkov Vitaly (MADE-DS-22)

Project Organization
------------

    ├── LICENSE
    ├── README.md               <- The top-level README for developers using this project.
    ├── configs                 <- YAML configs for the project
    │
    ├── data
    │   └── raw                 <- The original, immutable data dump.
    │
    ├── models                  <- Trained and serialized models, model predictions, metrics and transformers
    │
    ├── reports                 <- Generated Pandas Profiling of dataset
    │
    ├── requirements.txt        <- The requirements file for reproducing the analysis environment, e.g.
    │                               generated with `pip freeze > requirements.txt`
    │
    ├── setup.py                <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                     <- Source code for use in this project.
    │   ├── __init__.py         <- Makes src a Python module
    │   ├── build_features.py   <- Functions to turn raw data into features for modeling
    │   ├── models.py           <- Functions to train models and then use trained models to make
    │   │                          predictions
    │   ├── scaler.py           <- Custom Scaler class
    │   │
    ├── tests                   <- Code for testing
    │   ├── __init__.py         <- Makes tests a Python module
    │   ├── conftest.py         <- Fixtures for all tests modules
    │   ├── constants.py        <- Predefined constants
    │   ├── generate_data.py    <- Synthtic data generation function
    │   ├── test_e2e_eval.py    <- End-to-end evaluating pipeline test
    │   ├── test_e2e_training.py    <- End-to-end training pipeline test
    │   ├── test_unit_features.py   <- Unit testing of "build_features.py" module
    │   ├── test_unit_model.py    <- Unit testing of "models.py" module
    │
    ├── visualize.py       <- Function to create exploratory and results oriented visualizations (Pandas Profiling)
    ├── train_pipleine.py  <- Entry point for training the model
    ├── eval_pipleine.py   <- Entry point for evaluating the model
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------
Steps to build the project
------------
First of all, install necessary packages:
```
pip install -r requirements.txt
```
Make pandas profiling report, save it to `reports/`: 
```
python visualize.py
```

Train the model and save all artifacts to `models/`:
 ```
 python train_pipeline.py
```
Evaluate the model and save predictions to `models/`:
```
python eval_pipeline.py
```
Run all tests and check coverage:
```
pytest tests/ -v --cov -s 
```
You are awesome!
<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
