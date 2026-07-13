import requests
import pandas as pd

url = "https://api.open-meteo.com/v1/forecast"
params = {
    "latitude": 19.076,
    "longitude": 72.877,
    "hourly": "temperature_2m,relative_humidity_2m,precipitation,windspeed_10m,surface_pressure,cloudcover",
    "timezone": "Asia/Kolkata",
    "forecast_days": 3
}

response = requests.get(url, params=params)
data = response.json()

df = pd.DataFrame(data["hourly"])
df.to_csv("mumbai_weather_sample.csv", index=False)
print(df.head())
print(f"\nSaved {len(df)} rows to mumbai_weather_sample.csv")