# Created on April 19, 2025
# Author: MAITRI

import streamlit as st
import numpy as np
import pickle  


MODEL_PATH = "D:/machine learning/bostonhousing/bostonhousing1.sav"
with open(MODEL_PATH, 'rb') as f:
    model, scaler = pickle.load(f) 


st.set_page_config(page_title="🏡 Boston House Price Predictor", layout="centered")


st.markdown("""
    <style>
    .title {
        text-align: center;
        font-size: 2.5em;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 0.2em;
    }
    .subtitle {
        text-align: center;
        font-size: 1.2em;
        color: #7f8c8d;
        margin-bottom: 2rem;
    }
    .stButton > button {
        background-color: #27ae60;
        color: white;
        font-weight: bold;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
    }
    .stButton > button:hover {
        background-color: #1e8449;
        transition: 0.3s ease;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🏡 Boston House Price Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter housing details to predict the median value (in $1000s)</div>', unsafe_allow_html=True)


with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        crim = st.number_input("CRIM: Per capita crime rate", 0.0, 100.0, step=0.1)
        zn = st.number_input("ZN: Proportion of residential land zoned", 0.0, 100.0, step=0.1)
        indus = st.number_input("INDUS: Non-retail business acres", 0.0, 30.0, step=0.1)
        chas = st.number_input("CHAS: Bounds Charles River", )
        nox = st.number_input("NOX: Nitric oxide concentration", 0.0, 1.0, step=0.01)
        rm = st.number_input("RM: Avg. rooms per dwelling", 0.0, 10.0, step=0.1)
        age = st.number_input("AGE: % owner-occupied built before 1940", 0.0, 100.0, step=0.1)

    with col2:
        dis = st.number_input("DIS: Distance to employment centers", 0.0, 15.0, step=0.1)
        rad = st.number_input("RAD: Accessibility to highways", 1, 24, step=1)
        tax = st.number_input("TAX: Property tax rate", 100, 800, step=1)
        ptratio = st.number_input("PTRATIO: Pupil-teacher ratio", 10.0, 25.0, step=0.1)
        b = st.number_input("B: 1000(Bk - 0.63)^2", 0.0, 400.0, step=0.1)
        lstat = st.number_input("LSTAT: % lower status of population", 0.0, 40.0, step=0.1)

    submit = st.form_submit_button(" Predict")


if submit:
    try:
        chas = 1 if chas == "Yes" else 0
        input_data = np.array([[crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat]])
        scaled_input = scaler.transform(input_data)  
        prediction = model.predict(scaled_input)[0]

        st.success(f"🏠 Estimated Median House Value: **${prediction * 1000:.2f}**")
    except Exception as e:
        st.error(f" Error: {e}")

