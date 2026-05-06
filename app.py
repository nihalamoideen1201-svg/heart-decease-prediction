import os
from urllib.parse import urlparse

import requests
import streamlit as st


def normalize_backend_url(raw_url: str) -> str:
    value = raw_url.strip().rstrip("/")
    if not value:
        return "http://127.0.0.1:8000"

    parsed = urlparse(value)
    if not parsed.scheme:
        return f"http://{value}"

    return value


default_backend_url = normalize_backend_url(os.getenv("BACKEND_URL", "http://127.0.0.1:8000"))
backend_url_input = st.sidebar.text_input("Backend URL", value=default_backend_url)
BACKEND_URL = normalize_backend_url(backend_url_input)

st.title("Heart Disease Prediction System")
st.caption(f"Backend URL: {BACKEND_URL}")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex (1=Male, 0=Female)", [1, 0])
    cp = st.number_input("Chest Pain Type (0-3)", min_value=0, max_value=3, value=0)
    trestbps = st.number_input("Resting Blood Pressure", min_value=50, max_value=300, value=120)
    chol = st.number_input("Cholesterol", min_value=50, max_value=700, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])

with col2:
    restecg = st.number_input("Resting ECG Results (0-2)", min_value=0, max_value=2, value=0)
    thalach = st.number_input("Max Heart Rate", min_value=50, max_value=250, value=150)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=0.0)
    slope = st.number_input("Slope (0-2)", min_value=0, max_value=2, value=0)
    ca = st.number_input("Major Vessels (0-4)", min_value=0, max_value=4, value=0)
    thal = st.number_input("Thal (0-3)", min_value=0, max_value=3, value=0)

if st.button("Predict Heart Disease Status"):
    user_data = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,
    }

    try:
        response = requests.post(f"{BACKEND_URL}/predict", json=user_data, timeout=10)

        if response.status_code == 200:
            prediction = response.json()
            st.subheader("Results")
            st.metric("Predicted Class", prediction["risk_label"])
            if "probability" in prediction:
                st.metric("Predicted Probability", f'{prediction["probability"]:.2%}')

            if prediction["prediction"] == 1:
                st.error("The model predicts a high risk of heart disease.")
            else:
                st.success("The model predicts a low risk of heart disease.")
        else:
            st.error(f"Backend error: {response.status_code}")
            st.write(response.text)

    except Exception as exc:
        st.error(f"Could not connect to the backend. Error: {exc}")
        st.info("Start the API first with: python main.py")
