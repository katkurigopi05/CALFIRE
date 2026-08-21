"""
Fetch historical daily weather for every fire incident in California_Fire_Incidents.csv
using the free Open-Meteo Archive API (no key required).

This script must be run somewhere with open internet access (your own machine).
This sandbox's network policy blocks external API hosts, so it cannot run here.

Usage:
    pip install pandas requests
    python scripts/fetch_weather.py \
        --input California_Fire_Incidents.csv \
        --output weather_data.csv

Then upload the resulting weather_data.csv back into this project so it can be
merged with the fire records.

Output columns (one row per fire UniqueId):
    UniqueId, temp_max_c, temp_min_c, precip_mm, wind_max_kmh,
    et0_mm, humidity_mean_pct
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))
from model_common import clean_lat_lon

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
DAILY_VARS = "temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max,et0_fao_evapotranspiration"
MAX_RETRIES = 4


def fetch_one(lat, lon, date_str, session):
    """Fetch daily weather + mean relative humidity for a single lat/lon/date."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date_str,
        "end_date": date_str,
        "daily": DAILY_VARS,
        "hourly": "relative_humidity_2m",
        "timezone": "UTC",
    }
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(ARCHIVE_URL, params=params, timeout=20)
            if resp.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            data = resp.json()
            daily = data.get("daily", {})
            hourly_rh = data.get("hourly", {}).get("relative_humidity_2m")
            humidity_mean = None
            if hourly_rh:
                valid = [v for v in hourly_rh if v is not None]
                if valid:
                    humidity_mean = sum(valid) / len(valid)
            return {
                "temp_max_c": _first(daily.get("temperature_2m_max")),
                "temp_min_c": _first(daily.get("temperature_2m_min")),
                "precip_mm": _first(daily.get("precipitation_sum")),
                "wind_max_kmh": _first(daily.get("windspeed_10m_max")),
                "et0_mm": _first(daily.get("et0_fao_evapotranspiration")),
                "humidity_mean_pct": humidity_mean,
            }
        except requests.RequestException as exc:
            if attempt == MAX_RETRIES - 1:
                print(f"  failed for {lat},{lon},{date_str}: {exc}", file=sys.stderr)
                return None
            time.sleep(2 ** attempt)
    return None


def _first(values):
    return values[0] if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="California_Fire_Incidents.csv")
    parser.add_argument("--output", default="weather_data.csv")
    parser.add_argument("--sleep", type=float, default=0.3, help="seconds between requests")
    parser.add_argument("--limit", type=int, default=None, help="only process first N rows (for testing)")
    args = parser.parse_args()

    df = pd.read_csv(args.input, low_memory=False)
    df["Started"] = pd.to_datetime(df["Started"], errors="coerce", utc=True)
    df = clean_lat_lon(df)
    before = len(df)
    df = df.dropna(subset=["Latitude", "Longitude", "Started", "UniqueId"])
    print(f"Skipping {before - len(df)} rows with missing/invalid date or lat-lon")

    out_path = Path(args.output)
    done_ids = set()
    if out_path.exists():
        done_ids = set(pd.read_csv(out_path)["UniqueId"])
        print(f"Resuming: {len(done_ids)} rows already fetched")

    todo = df[~df["UniqueId"].isin(done_ids)]
    if args.limit:
        todo = todo.head(args.limit)

    print(f"Fetching weather for {len(todo)} incidents...")
    session = requests.Session()
    write_header = not out_path.exists()

    with open(out_path, "a", newline="") as f:
        for i, (_, row) in enumerate(todo.iterrows(), 1):
            date_str = row["Started"].strftime("%Y-%m-%d")
            result = fetch_one(row["Latitude"], row["Longitude"], date_str, session)
            record = {"UniqueId": row["UniqueId"], **(result or {})}
            pd.DataFrame([record]).to_csv(f, header=write_header, index=False)
            write_header = False
            f.flush()
            if i % 25 == 0:
                print(f"  {i}/{len(todo)} done")
            time.sleep(args.sleep)

    print(f"Done. Weather data saved to {out_path}")


if __name__ == "__main__":
    main()
