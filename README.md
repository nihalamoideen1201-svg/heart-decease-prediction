# heart-disease-prediction

Heart disease risk prediction demo built with FastAPI, Streamlit, and a saved scikit-learn pipeline.

## What changed

- Added request validation and safer API error handling.
- Added `/health` for deployment health checks.
- Added probability output when the model supports `predict_proba`.
- Added repeatable evaluation in `evaluate_model.py`.
- Added API tests in `tests/test_api.py`.
- Added a `Dockerfile` for container deployment.

## Run locally

1. Install dependencies with `pip install -r requirements.txt`
2. Start the FastAPI backend with `python main.py`
3. Start the Streamlit frontend with `streamlit run app.py`

## Test

Run `python -m unittest discover -s tests`

## Evaluate the model

The notebook trained from a local CSV outside the repository. To reproduce evaluation, place the dataset on disk and run:

```bash
python evaluate_model.py --data path/to/heart.csv
```

This writes `evaluation_report.json` with:

- accuracy
- ROC AUC when available
- confusion matrix
- classification report
- 5-fold cross-validation accuracy

## Docker

Build and run:

```bash
docker build -t heart-disease-api .
docker run -p 8000:8000 heart-disease-api
```

The API will be available at `http://localhost:8000` and health checks at `http://localhost:8000/health`.
