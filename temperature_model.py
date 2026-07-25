import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("mumbai_historical_weather.csv", parse_dates=["time"])
df = df.sort_values("time").reset_index(drop=True)

df["month"] = df["time"].dt.month
df["day_of_year"] = df["time"].dt.dayofyear
df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
df["is_monsoon"] = df["month"].isin([6, 7, 8, 9]).astype(int)
df["temp_lag1"] = df["temperature_2m_max"].shift(1)
df["temp_lag7"] = df["temperature_2m_max"].shift(7)
df["temp_roll7"] = df["temperature_2m_max"].shift(1).rolling(7).mean()
df = df.dropna().reset_index(drop=True)

features = ["precipitation_sum", "windspeed_10m_max", "relative_humidity_2m_mean",
            "surface_pressure_mean", "doy_sin", "doy_cos", "is_monsoon",
            "temp_lag1", "temp_lag7", "temp_roll7"]

X = df[features]
y = df["temperature_2m_max"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

rf = RandomForestRegressor(n_estimators=300, random_state=42)
rf.fit(X_train, y_train)
pred = rf.predict(X_test)

mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)
print(f"Temperature model: MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.3f}")

importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
print("\nFeature importances:\n", importances)