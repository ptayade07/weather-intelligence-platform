import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("mumbai_historical_weather.csv", parse_dates=["time"])
df = df.sort_values("time").reset_index(drop=True)

df["month"] = df["time"].dt.month
df["day_of_year"] = df["time"].dt.dayofyear
df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
df["is_monsoon"] = df["month"].isin([6, 7, 8, 9]).astype(int)
df["precip_lag1"] = df["precipitation_sum"].shift(1)
df["precip_lag7"] = df["precipitation_sum"].shift(7)
df["precip_roll7"] = df["precipitation_sum"].shift(1).rolling(7).mean()
df = df.dropna().reset_index(drop=True)

features = ["temperature_2m_max", "temperature_2m_min", "windspeed_10m_max",
            "relative_humidity_2m_mean", "surface_pressure_mean",
            "doy_sin", "doy_cos", "is_monsoon",
            "precip_lag1", "precip_lag7", "precip_roll7"]

X = df[features]
y = df["precipitation_sum"]

model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X, y)

joblib.dump({"model": model, "features": features}, "precip_model.joblib")
print("Saved precip_model.joblib")