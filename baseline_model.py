import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

df = pd.read_csv("mumbai_historical_weather.csv", parse_dates=["time"])

# Correlation analysis
numeric_cols = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum",
                 "windspeed_10m_max", "relative_humidity_2m_mean", "surface_pressure_mean"]
corr = df[numeric_cols].corr()
print("Correlation with precipitation_sum:\n", corr["precipitation_sum"].sort_values(ascending=False))

# Features / target
X = df[["temperature_2m_max", "temperature_2m_min", "windspeed_10m_max",
        "relative_humidity_2m_mean", "surface_pressure_mean"]]
y = df["precipitation_sum"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Baseline 1: Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)

# Baseline 2: Random Forest
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)

for name, pred in [("Linear Regression", pred_lr), ("Random Forest", pred_rf)]:
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    print(f"\n{name}: MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.3f}")