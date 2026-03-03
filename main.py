from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello Nihala"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str | None = None):
    return {"item_id":5, "q": "somequerry"}



from fastapi import FastAPI
import joblib
from pydantic import BaseModel

# 1. Initialize the app
app = FastAPI()

# 2. Load your pickled model 
# IMPORTANT: Ensure your model file is named 'model.pkl' and is in C:\Users\DELL\
model = joblib.load("model.pkl")

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
@app.post("/predict")
def predict(data: ModelInput):
    # Map the 13 input fields into a list for the model
    input_features = [[
        data.age, data.sex, data.cp, data.trestbps, data.chol, 
        data.fbs, data.restecg, data.thalach, data.exang, 
        data.oldpeak, data.slope, data.ca, data.thal
    ]]
    
    # Run prediction using the loaded joblib model
    prediction = model.predict(input_features)
    
    return {"prediction": int(prediction[0])}