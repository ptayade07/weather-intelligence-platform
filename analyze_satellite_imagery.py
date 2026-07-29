import cv2
import numpy as np
import pandas as pd
import glob
import os

def estimate_cloud_coverage(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Clouds: high brightness (V) and low saturation (S) - i.e. white/grey pixels
    lower = np.array([0, 0, 180])
    upper = np.array([180, 60, 255])
    mask = cv2.inRange(hsv, lower, upper)
    cloud_pixels = np.count_nonzero(mask)
    total_pixels = mask.size
    return (cloud_pixels / total_pixels) * 100

results = []
for path in sorted(glob.glob("satellite_images/*.jpg")):
    date = os.path.basename(path).replace("mumbai_", "").replace(".jpg", "")
    pct = estimate_cloud_coverage(path)
    results.append({"date": date, "estimated_cloud_pct": pct})
    print(f"{date}: estimated cloud coverage = {pct:.1f}%")

df = pd.DataFrame(results)
df.to_csv("satellite_cloud_estimates.csv", index=False)
print("\nSaved satellite_cloud_estimates.csv")

# Compare against recorded weather data for the same dates, if available
try:
    weather = pd.read_csv("mumbai_historical_weather.csv", parse_dates=["time"])
    weather["date_str"] = weather["time"].dt.strftime("%Y-%m-%d")
    merged = df.merge(weather, left_on="date", right_on="date_str", how="left")
    cols_to_show = ["date", "estimated_cloud_pct"]
    for c in ["cloudcover_mean", "precipitation_sum", "relative_humidity_2m_mean"]:
        if c in merged.columns:
            cols_to_show.append(c)
    print("\nComparison against recorded weather data:")
    print(merged[cols_to_show])
except Exception as e:
    print(f"\nCould not merge with weather data: {e}")