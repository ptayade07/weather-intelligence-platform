import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

rf = RandomForestRegressor(n_estimators=300, random_state=42)
rf.fit(X_train, y_train)

explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=150)
print("Saved shap_summary.png")

mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=features).sort_values(ascending=False)
print("\nMean |SHAP value| per feature:\n", mean_abs_shap)