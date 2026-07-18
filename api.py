import numpy as np
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

bundle = joblib.load("precip_model.joblib")
model = bundle["model"]
features = bundle["features"]

app = FastAPI(title="Monsoon Precipitation Predictor")

class WeatherInput(BaseModel):
    temperature_2m_max: float
    temperature_2m_min: float
    windspeed_10m_max: float
    relative_humidity_2m_mean: float
    surface_pressure_mean: float
    day_of_year: int
    is_monsoon: int
    precip_lag1: float
    precip_lag7: float
    precip_roll7: float

@app.get("/")
def root():
    return {"status": "ok", "model": "RandomForest precipitation predictor"}

@app.post("/predict")
def predict(data: WeatherInput):
    doy_sin = np.sin(2 * np.pi * data.day_of_year / 365)
    doy_cos = np.cos(2 * np.pi * data.day_of_year / 365)
    row = [[
        data.temperature_2m_max, data.temperature_2m_min, data.windspeed_10m_max,
        data.relative_humidity_2m_mean, data.surface_pressure_mean,
        doy_sin, doy_cos, data.is_monsoon,
        data.precip_lag1, data.precip_lag7, data.precip_roll7
    ]]
    pred = model.predict(row)[0]
    return {"predicted_precipitation_mm": round(float(pred), 2)}