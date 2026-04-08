from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "model.pkl"

FEATURE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

app = FastAPI(
    title="Heart Disease Prediction API",
    version="1.0.0",
    description="Predicts heart disease risk from structured clinical features.",
)


class ModelInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age: int = Field(..., ge=1, le=120)
    sex: int = Field(..., ge=0, le=1)
    cp: int = Field(..., ge=0, le=3)
    trestbps: int = Field(..., ge=50, le=300)
    chol: int = Field(..., ge=50, le=700)
    fbs: int = Field(..., ge=0, le=1)
    restecg: int = Field(..., ge=0, le=2)
    thalach: int = Field(..., ge=50, le=250)
    exang: int = Field(..., ge=0, le=1)
    oldpeak: float = Field(..., ge=0, le=10)
    slope: int = Field(..., ge=0, le=2)
    ca: int = Field(..., ge=0, le=4)
    thal: int = Field(..., ge=0, le=3)


@lru_cache(maxsize=1)
def load_model() -> Any:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def build_features(data: ModelInput) -> pd.DataFrame:
    return pd.DataFrame([[getattr(data, column) for column in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)


@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "Heart Disease Prediction API is running."}


@app.get("/health")
def health_check() -> dict[str, str]:
    try:
        load_model()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model unavailable: {exc}") from exc

    return {"status": "ok", "model_path": str(MODEL_PATH.name)}


@app.post("/predict")
def predict(data: ModelInput) -> dict[str, Any]:
    try:
        model = load_model()
        input_features = build_features(data)
        prediction = int(model.predict(input_features)[0])
        result: dict[str, Any] = {
            "prediction": prediction,
            "risk_label": "high_risk" if prediction == 1 else "low_risk",
        }

        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(input_features)[0][1])
            result["probability"] = round(probability, 4)

        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
