import os

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

st.title("Heart Disease Prediction System")
st.caption(f"Backend URL: {BACKEND_URL}")

# Create two columns for a cleaner layout
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", value=50)
    sex = st.selectbox("Sex (1=Male, 0=Female)", [1, 0])
    cp = st.number_input("Chest Pain Type (0-3)", value=0)
    trestbps = st.number_input("Resting Blood Pressure", value=120)
    chol = st.number_input("Cholesterol", value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])

with col2:
    restecg = st.number_input("Resting ECG results", value=0)
    thalach = st.number_input("Max Heart Rate", value=150)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("ST Depression", value=0.0)
    slope = st.number_input("Slope", value=0)
    ca = st.number_input("Major Vessels", value=0)
    thal = st.number_input("Thal", value=0)

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
            st.subheader("Results:")
            if prediction["prediction"] == 1:
                st.error("The model predicts a high risk of heart disease.")
            else:
                st.success("The model predicts a low risk of heart disease.")
        else:
            st.error(f"Backend error: {response.status_code}")
            st.write(response.text)

    except Exception as e:
        st.error(f"Could not connect to the backend. Error: {e}")
        st.info("Start the API first with: python main.py")
