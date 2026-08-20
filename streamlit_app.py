import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

st.set_page_config(page_title="Mumbai Monsoon Forecast", page_icon="🌧️", layout="centered")

st.title("🌧️ Mumbai Monsoon Precipitation Forecast")
st.caption("Weather Intelligence & Climate Decision Support Platform — Phase 02, MacroEdtech GenAI Research Internship")

bundle = joblib.load("precip_model.joblib")
model = bundle["model"]
feature_order = bundle["features"]

st.subheader("Enter today's conditions")

col1, col2 = st.columns(2)
with col1:
    temp_max = st.slider("Max Temperature (°C)", 18.0, 42.0, 30.0)
    temp_min = st.slider("Min Temperature (°C)", 15.0, 32.0, 24.0)
    windspeed = st.slider("Wind Speed (km/h)", 0.0, 40.0, 15.0)
with col2:
    humidity = st.slider("Relative Humidity (%)", 10.0, 100.0, 80.0)
    pressure = st.slider("Surface Pressure (hPa)", 990.0, 1020.0, 1005.0)

st.subheader("Recent rainfall (mm)")
col3, col4, col5 = st.columns(3)
with col3:
    precip_lag1 = st.number_input("Yesterday", 0.0, 300.0, 10.0)
with col4:
    precip_lag7 = st.number_input("7 days ago", 0.0, 300.0, 5.0)
with col5:
    precip_roll7 = st.number_input("7-day average", 0.0, 300.0, 8.0)

if st.button("Predict Precipitation", type="primary"):
    doy = datetime.now().timetuple().tm_yday
    doy_sin = np.sin(2 * np.pi * doy / 365)
    doy_cos = np.cos(2 * np.pi * doy / 365)
    is_monsoon = 1 if datetime.now().month in [6, 7, 8, 9] else 0

    X = pd.DataFrame([{
        "temperature_2m_max": temp_max,
        "temperature_2m_min": temp_min,
        "windspeed_10m_max": windspeed,
        "relative_humidity_2m_mean": humidity,
        "surface_pressure_mean": pressure,
        "doy_sin": doy_sin,
        "doy_cos": doy_cos,
        "is_monsoon": is_monsoon,
        "precip_lag1": precip_lag1,
        "precip_lag7": precip_lag7,
        "precip_roll7": precip_roll7,
    }])[feature_order]

    pred = model.predict(X)[0]
    st.metric("Predicted precipitation (next day)", f"{pred:.2f} mm")

    if pred < 2.5:
        st.info("Light or no rain expected.")
    elif pred < 15:
        st.warning("Moderate rain expected.")
    else:
        st.error("Heavy rain expected — flood/disruption risk.")

with st.expander("Model info"):
    st.write(f"Features used: {feature_order}")
    st.write("Model: Random Forest (MAE = 3.44mm, R² = 0.669 on held-out test data)")