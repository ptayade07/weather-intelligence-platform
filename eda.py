import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("mumbai_historical_weather.csv", parse_dates=["time"])
df["year"] = df["time"].dt.year
df["month"] = df["time"].dt.month

print(df.describe())
print("\nMissing values:\n", df.isna().sum())

# Monsoon months only (June-Sept)
monsoon = df[df["month"].isin([6,7,8,9])]
monthly_rain = monsoon.groupby(["year","month"])["precipitation_sum"].sum().reset_index()

plt.figure(figsize=(10,5))
for m in [6,7,8,9]:
    subset = monthly_rain[monthly_rain["month"]==m]
    plt.plot(subset["year"], subset["precipitation_sum"], label=f"Month {m}")
plt.legend()
plt.title("Monthly Monsoon Precipitation by Year - Mumbai")
plt.xlabel("Year")
plt.ylabel("Total Precipitation (mm)")
plt.savefig("monsoon_precipitation_trend.png")
print("\nSaved plot: monsoon_precipitation_trend.png")