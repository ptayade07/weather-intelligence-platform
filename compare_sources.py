import pandas as pd

om = pd.read_csv("mumbai_historical_weather.csv", parse_dates=["time"])
nasa = pd.read_csv("mumbai_nasa_power.csv", parse_dates=["date"])

merged = om.merge(nasa, left_on="time", right_on="date")
print("Temperature correlation (Open-Meteo vs NASA POWER):", merged["temperature_2m_max"].corr(merged["temp_max_nasa"]))
print("Precipitation correlation (Open-Meteo vs NASA POWER):", merged["precipitation_sum"].corr(merged["precip_nasa"]))
print("Temp mean diff:", (merged["temperature_2m_max"] - merged["temp_max_nasa"]).mean())
print("Precip mean diff:", (merged["precipitation_sum"] - merged["precip_nasa"]).mean())