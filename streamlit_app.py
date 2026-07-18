import streamlit as st
import requests

st.title("Mumbai Monsoon Precipitation Predictor")

with st.form("predict_form"):
    temp_max = st.number_input("Max Temperature (°C)", value=30.0)
    temp_min = st.number_input("Min Temperature (°C)", value=24.0)
    wind = st.number_input("Max Wind Speed (km/h)", value=15.0)
    humidity = st.number_input("Avg Humidity (%)", value=80.0)
    pressure = st.number_input("Avg Surface Pressure (hPa)", value=1006.0)
    day_of_year = st.number_input("Day of Year (1-365)", value=200, min_value=1, max_value=365)
    is_monsoon = st.selectbox("Monsoon Month?", [1, 0])
    lag1 = st.number_input("Yesterday's Precipitation (mm)", value=5.0)
    lag7 = st.number_input("Precipitation 7 Days Ago (mm)", value=5.0)
    roll7 = st.number_input("7-Day Avg Precipitation (mm)", value=5.0)
    submitted = st.form_submit_button("Predict")

if submitted:
    payload = {
        "temperature_2m_max": temp_max, "temperature_2m_min": temp_min,
        "windspeed_10m_max": wind, "relative_humidity_2m_mean": humidity,
        "surface_pressure_mean": pressure, "day_of_year": int(day_of_year),
        "is_monsoon": int(is_monsoon), "precip_lag1": lag1,
        "precip_lag7": lag7, "precip_roll7": roll7
    }
    response = requests.post("http://127.0.0.1:8000/predict", json=payload)
    result = response.json()
    st.success(f"Predicted precipitation: {result['predicted_precipitation_mm']} mm")