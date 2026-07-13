import requests
import pandas as pd

url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 19.076,
    "longitude": 72.877,
    "start_date": "2015-06-01",
    "end_date": "2025-09-30",
    "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max,relative_humidity_2m_mean,surface_pressure_mean",
    "timezone": "Asia/Kolkata"
}

response = requests.get(url, params=params)
data = response.json()

df = pd.DataFrame(data["daily"])
df.to_csv("mumbai_historical_weather.csv", index=False)
print(df.shape)
print(df.head())