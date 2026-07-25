import requests
import pandas as pd

url = "https://power.larc.nasa.gov/api/temporal/daily/point"
params = {
    "parameters": "T2M_MAX,PRECTOTCORR,RH2M,WS10M",
    "community": "AG",
    "longitude": 72.8777,
    "latitude": 19.0760,
    "start": "20150101",
    "end": "20251231",
    "format": "JSON",
}

resp = requests.get(url, params=params)
resp.raise_for_status()
data = resp.json()["properties"]["parameter"]

df = pd.DataFrame({
    "date": list(data["T2M_MAX"].keys()),
    "temp_max_nasa": list(data["T2M_MAX"].values()),
    "precip_nasa": list(data["PRECTOTCORR"].values()),
    "humidity_nasa": list(data["RH2M"].values()),
    "windspeed_nasa": list(data["WS10M"].values()),
})
df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
df.to_csv("mumbai_nasa_power.csv", index=False)
print(f"Saved {len(df)} rows to mumbai_nasa_power.csv")
print(df.head())
print(df.describe())