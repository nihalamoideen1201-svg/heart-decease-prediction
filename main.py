
from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

# 1. Initialize the app
app = FastAPI()

# 2. Load your pickled model 
# IMPORTANT: Ensure your model file is named 'model.pkl' and is in C:\Users\DELL\
model = joblib.load("model.pkl")

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

# 3. Define the structure for your model inputs
class ModelInput(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

@app.get("/")
def read_root():
    return {"message": "Hello Nihala, the model is ready!"}

@app.post("/predict")
def predict(data: ModelInput):
    # Preserve feature names expected by the trained sklearn pipeline.
    input_features = pd.DataFrame(
        [[
            data.age,
            data.sex,
            data.cp,
            data.trestbps,
            data.chol,
            data.fbs,
            data.restecg,
            data.thalach,
            data.exang,
            data.oldpeak,
            data.slope,
            data.ca,
            data.thal,
        ]],
        columns=FEATURE_COLUMNS,
    )
    
    # Run prediction using the loaded joblib model
    prediction = model.predict(input_features)
    
    return {"prediction": int(prediction[0])}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
