# heart-disease-prediction

Heart disease risk prediction demo built with FastAPI, Streamlit, and a saved scikit-learn pipeline.

## Project layout

- `main.py`: FastAPI backend with `/health` and `/predict`
- `app.py`: Streamlit frontend
- `train_model.py`: reproducible model training and MLflow logging
- `evaluate_model.py`: repeatable offline evaluation against a CSV dataset
- `check_dataset.py`: quick dataset inspection helper
- `retrain_if_needed.py`: controlled retraining when `data/heart.csv` changes
- `training_report.json`: metrics from the latest training run saved by `train_model.py`

## Dataset location

The dataset is not stored in this repository. Put the CSV anywhere on your machine and pass its path explicitly with `--data`.

Examples:

```bash
python check_dataset.py --data path/to/heart.csv
python train_model.py --data path/to/heart.csv
python evaluate_model.py --data path/to/heart.csv
python retrain_if_needed.py --data path/to/heart.csv
```

The CSV is expected to contain the model target column, which defaults to `target`.

## Install

```bash
pip install -r requirements.txt
```

## Run locally

Start the backend:

```bash
python main.py
```

Start the frontend in a second terminal:

```bash
streamlit run app.py
```

By default the frontend calls `http://127.0.0.1:8000`. You can also set `BACKEND_URL` before launching Streamlit or change the backend URL in the Streamlit sidebar.

## Train

Train a new model and write both `model.pkl` and `training_report.json`:

```bash
python train_model.py --data path/to/heart.csv --model model.pkl --target target --random-state 42
```

Optional MLflow flags:

```bash
python train_model.py --data path/to/heart.csv --experiment-name heart-disease-prediction --run-name baseline
```

Artifacts:

- `model.pkl`: trained scikit-learn pipeline used by the API
- `training_report.json`: JSON summary of the training run
- `mlruns/`: local MLflow tracking data

## Controlled retraining

Retrain only when the dataset changes:

```bash
python retrain_if_needed.py --data data/heart.csv
```

What this does:

- validates the expected schema before training
- hashes the dataset and compares it with `model_metadata.json`
- trains a candidate model instead of replacing the current one immediately
- promotes the candidate only if it passes minimum metric thresholds
- blocks regressions by default when an existing model metadata file is present

Useful flags:

```bash
python retrain_if_needed.py --data data/heart.csv --force
python retrain_if_needed.py --data data/heart.csv --min-accuracy 0.80 --min-roc-auc 0.85
python retrain_if_needed.py --data data/heart.csv --allow-regression
```

Metadata:

- `model_metadata.json`: stores dataset hash, row count, model path, feature columns, and accepted metrics

## Evaluate

Run offline evaluation against a dataset and save a report:

```bash
python evaluate_model.py --data path/to/heart.csv --model model.pkl --output evaluation_report.json
```

This writes:

- `accuracy`
- `roc_auc` when the model supports probabilities
- `confusion_matrix`
- `classification_report`
- `cross_validation_accuracy`

## training_report.json

`training_report.json` is produced by `train_model.py` from the held-out test split plus 5-fold cross-validation. It contains:

- `accuracy`: test split accuracy
- `confusion_matrix`: counts of true/false positives and negatives
- `classification_report`: per-class precision, recall, f1-score, and support
- `roc_auc`: ranking quality for the positive class when probabilities are available
- `cross_validation_accuracy`: mean, standard deviation, and per-fold accuracy scores

## Test

Run:

```bash
python -m unittest discover -s tests
```

The test suite covers successful predictions, validation failures, missing model behavior, probability field behavior, and health-check failure handling.

## Docker

Build:

```bash
docker build -t heart-disease-api .
```

Run:

```bash
docker run -p 8000:8000 heart-disease-api
```

If your platform injects a port dynamically, the container honors the `PORT` environment variable.

## Deployment note

For a submission or demo, the current app shape is enough once the repository is clean and the model artifact is present.

For real use, the next gaps are:

- model versioning beyond local artifact replacement
- published input schema and feature documentation
- stronger validation and monitoring around prediction inputs
