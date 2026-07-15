import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("mumbai_historical_weather.csv", parse_dates=["time"])
df = df.sort_values("time").reset_index(drop=True)

df["month"] = df["time"].dt.month
df["day_of_year"] = df["time"].dt.dayofyear
df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
df["is_monsoon"] = df["month"].isin([6, 7, 8, 9]).astype(int)

exog_cols = ["temperature_2m_max", "temperature_2m_min", "windspeed_10m_max",
             "relative_humidity_2m_mean", "surface_pressure_mean",
             "doy_sin", "doy_cos", "is_monsoon"]

y = df["precipitation_sum"]
X = df[exog_cols]

split_idx = int(len(df) * 0.8)
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]

model = SARIMAX(y_train, exog=X_train, order=(2, 0, 1),
                 enforce_stationarity=False, enforce_invertibility=False)
fit = model.fit(disp=False)

pred = fit.forecast(steps=len(y_test), exog=X_test)
pred = np.clip(pred, 0, None)

mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)
print(f"SARIMAX: MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.3f}")