import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("mumbai_historical_weather.csv", parse_dates=["time"])
df = df.sort_values("time").reset_index(drop=True)

# Prophet expects columns named 'ds' (date) and 'y' (target)
prophet_df = df[["time", "precipitation_sum"]].rename(columns={"time": "ds", "precipitation_sum": "y"})

# Chronological split (last 20% as test, matching Update 4's approach)
split_idx = int(len(prophet_df) * 0.8)
train_df = prophet_df.iloc[:split_idx]
test_df = prophet_df.iloc[split_idx:]

model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
model.fit(train_df)

future = test_df[["ds"]]
forecast = model.predict(future)

y_true = test_df["y"].values
y_pred = forecast["yhat"].values
y_pred = np.clip(y_pred, 0, None)  # precipitation can't be negative

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

print(f"Prophet: MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.3f}")