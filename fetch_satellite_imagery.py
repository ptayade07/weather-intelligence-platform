import requests
import os

os.makedirs("satellite_images", exist_ok=True)

# Mumbai bounding box (lon_min, lat_min, lon_max, lat_max)
bbox = "72.6,18.8,73.1,19.3"

# A handful of monsoon-season dates to pull true-color imagery for
dates = ["2024-07-15", "2024-08-10", "2024-06-05", "2024-01-15", "2024-12-01"]

for date in dates:
    url = (
        "https://wvs.earthdata.nasa.gov/api/v1/snapshot"
        f"?REQUEST=GetSnapshot&TIME={date}"
        "&BBOX=" + bbox +
        "&CRS=EPSG:4326"
        "&LAYERS=MODIS_Terra_CorrectedReflectance_TrueColor"
        "&FORMAT=image/jpeg"
        "&WIDTH=512&HEIGHT=512"
    )
    resp = requests.get(url, timeout=30)
    if resp.status_code == 200 and len(resp.content) > 1000:
        path = f"satellite_images/mumbai_{date}.jpg"
        with open(path, "wb") as f:
            f.write(resp.content)
        print(f"Saved {path} ({len(resp.content)} bytes)")
    else:
        print(f"Failed for {date}: status={resp.status_code}, len={len(resp.content)}")