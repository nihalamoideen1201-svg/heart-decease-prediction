# from fastapi import FastAPI
# import joblib
# from pydantic import BaseModel


import os
import streamlit as st
import requests
BACKEND_URL = "http://54.252.223.57:8000"
st.title("❤️ Heart Disease Prediction System")

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

if st.button("Predict"):
    input_data = {
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
    }
    
    # This sends the data to your FastAPI running in the other terminal
    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=input_data)
        if response.status_code == 200:
            result = response.json()
            st.success(f"The Prediction is: {result}")
        else:
            st.error(f"API Error: {response.status_code}")
    except Exception as e:
        st.error(f"Connection failed: Ensure FastAPI is running on port 8000")

         

if st.button("Predict Heart Disease Status"):
    
    # 2. Gather all the input variables into a dictionary
    user_data = {
    "age": age,
    "sex": sex,
    "cp": cp,
    "trestbps": trestbps,
    "chol": chol,
    "fbs": fbs,
    "restecg": restecg,
    "thalach": thalach,  # Add these based on your col2 variable names
    "exang": exang,
    "oldpeak": oldpeak,
    "slope": slope,
    "ca": ca,
    "thal": thal
}

    # 3. Send data to FastAPI (Make sure main.py is running on port 8000)
    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=user_data)
        prediction = response.json()
        
        # 4. Display the result
        if prediction['prediction'] == 1:
            st.error("The model predicts a high risk of heart disease.")
        else:
            st.success("The model predicts a low risk of heart disease.")
            
    except Exception as e:
        st.error(f"Backend connection failed. Is your FastAPI server running? Error: {e}")

