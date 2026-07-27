from mcp.server.fastmcp import FastMCP
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

mcp = FastMCP("weather-tools")

model_bundle = joblib.load("precip_model.joblib")
model = model_bundle["model"]
feature_order = model_bundle["features"]

@mcp.tool()
def predict_precipitation(temperature_max: float, temperature_min: float,
                           windspeed_max: float, relative_humidity: float,
                           surface_pressure: float, precip_lag1: float,
                           precip_lag7: float, precip_roll7: float) -> str:
    """Predict next-day precipitation (mm) for Mumbai given current weather conditions."""
    doy = datetime.now().timetuple().tm_yday
    doy_sin = np.sin(2 * np.pi * doy / 365)
    doy_cos = np.cos(2 * np.pi * doy / 365)
    is_monsoon = 1 if datetime.now().month in [6, 7, 8, 9] else 0

    X = pd.DataFrame([{
        "temperature_2m_max": temperature_max,
        "temperature_2m_min": temperature_min,
        "windspeed_10m_max": windspeed_max,
        "relative_humidity_2m_mean": relative_humidity,
        "surface_pressure_mean": surface_pressure,
        "doy_sin": doy_sin,
        "doy_cos": doy_cos,
        "is_monsoon": is_monsoon,
        "precip_lag1": precip_lag1,
        "precip_lag7": precip_lag7,
        "precip_roll7": precip_roll7,
    }])
    pred = model.predict(X[feature_order])[0]
    return f"Predicted precipitation: {pred:.2f} mm"

if __name__ == "__main__":
    mcp.run(transport="stdio")