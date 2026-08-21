"""
CALFIRE real-time fire-risk tool.

Scores a hypothetical or upcoming fire (county + date, optionally exact
location and weather) against every acreage threshold trained by
scripts/threshold_analysis.py (10/50/100/500/1,000 acres), printing a full
risk profile instead of a single yes/no call.

Weather can be supplied two ways:
  1. Manually, via --temp-max/--temp-min/--precip/--wind-max/--et0/--humidity
     (use whatever your own weather source gives you for that day/location).
  2. Live, via --fetch-weather, which calls the same Open-Meteo API as
     scripts/fetch_weather.py. This requires outbound internet access — it
     will not work inside a sandboxed environment with restricted egress,
     only wherever you actually run this with a live network path.
If no weather is given at all, the model falls back to its imputed median
for those features (accuracy suffers accordingly — weather is a real driver
of fire spread).

Usage:
    python scripts/predict.py --county Riverside --date 2026-08-15
    python scripts/predict.py --county Butte --date 2026-07-01 --fetch-weather
    python scripts/predict.py --county Kern --date 2026-09-01 \\
        --temp-max 41 --wind-max 45 --humidity 12 --precip 0
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).parent.parent
MULTI_MODEL_PATH = ROOT / "output" / "multi_threshold_models.pkl"

RISK_TIERS = [
    (0.0, 0.25, "LOW"),
    (0.25, 0.5, "MODERATE"),
    (0.5, 0.75, "HIGH"),
    (0.75, 1.01, "CRITICAL"),
]


def risk_tier(prob):
    for lo, hi, name in RISK_TIERS:
        if lo <= prob < hi:
            return name
    return "CRITICAL"


def try_live_weather(lat, lon, date_str):
    sys.path.insert(0, str(Path(__file__).parent))
    from fetch_weather import fetch_one
    import requests
    try:
        result = fetch_one(lat, lon, date_str, requests.Session())
        if result is None:
            print("  Live weather fetch failed (no network path to weather API from here).", file=sys.stderr)
        return result or {}
    except Exception as exc:
        print(f"  Live weather fetch failed: {exc}", file=sys.stderr)
        return {}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--county", required=True, help="e.g. Riverside")
    parser.add_argument("--date", required=True, help="YYYY-MM-DD, the ignition/assessment date")
    parser.add_argument("--lat", type=float, default=None)
    parser.add_argument("--lon", type=float, default=None)
    parser.add_argument("--fetch-weather", action="store_true", help="attempt a live Open-Meteo lookup")
    parser.add_argument("--temp-max", type=float, default=None, help="max temperature, C")
    parser.add_argument("--temp-min", type=float, default=None, help="min temperature, C")
    parser.add_argument("--precip", type=float, default=None, help="precipitation, mm")
    parser.add_argument("--wind-max", type=float, default=None, help="max wind speed, km/h")
    parser.add_argument("--et0", type=float, default=None, help="reference evapotranspiration, mm")
    parser.add_argument("--humidity", type=float, default=None, help="mean relative humidity, %%")
    args = parser.parse_args()

    if not MULTI_MODEL_PATH.exists():
        raise FileNotFoundError(f"{MULTI_MODEL_PATH} not found. Run scripts/threshold_analysis.py first.")
    bundle = joblib.load(MULTI_MODEL_PATH)
    thresholds = bundle["thresholds"]

    county = args.county.strip()
    date = pd.to_datetime(args.date, utc=True)

    lat, lon = args.lat, args.lon
    if lat is None or lon is None:
        centroid = bundle["county_centroid_lookup"].get(county)
        if centroid is None:
            raise ValueError(
                f"Unknown county '{county}' and no --lat/--lon given. "
                f"Known counties: {sorted(bundle['county_centroid_lookup'].keys())}"
            )
        lat, lon = centroid
        print(f"Using historical centroid for {county}: ({lat:.3f}, {lon:.3f})")

    weather = {
        "temp_max_c": args.temp_max, "temp_min_c": args.temp_min,
        "precip_mm": args.precip, "wind_max_kmh": args.wind_max,
        "et0_mm": args.et0, "humidity_mean_pct": args.humidity,
    }
    if args.fetch_weather:
        print(f"Fetching live weather for ({lat:.3f}, {lon:.3f}) on {args.date}...")
        live = try_live_weather(lat, lon, args.date)
        for k, v in live.items():
            if weather.get(k) is None:
                weather[k] = v

    doy = date.dayofyear
    base_row = {
        "StartYear": date.year,
        "DOY_sin": np.sin(2 * np.pi * doy / 365.25),
        "DOY_cos": np.cos(2 * np.pi * doy / 365.25),
        "StartWeekday": date.dayofweek,
        "DrySeason": int(6 <= date.month <= 10),
        "Status": "Active",
        "Latitude": lat,
        "Longitude": lon,
        **weather,
    }

    print("\n" + "=" * 66)
    print(f"CALFIRE Risk Profile — {county} County, {args.date}")
    print("=" * 66)
    print(f"{'Threshold':>12} | {'Probability':>11} | {'Tier':>9} | {'Call':>26}")
    print("-" * 66)

    for t, info in sorted(thresholds.items()):
        row = dict(base_row)
        row["CountyHistoricalRate"] = info["county_rate_lookup"].get(county, info["global_rate"])
        X = pd.DataFrame([row])[info["numeric"] + info["categorical"] + info["geo"]]
        proba = info["pipeline"].predict_proba(X)[0, 1]
        tier = risk_tier(proba)
        call = f">= {t:,} ac" if proba >= info["threshold_cutoff"] else f"< {t:,} ac (likely)"
        print(f"{t:>10,} ac | {proba:>10.1%} | {tier:>9} | {call:>26}")

    print("-" * 66)
    model_summary = ", ".join(f"{t}ac={info['model_name']}" for t, info in sorted(thresholds.items()))
    print(f"Models: {model_summary}")

    missing_weather = [k for k, v in weather.items() if v is None]
    if missing_weather:
        print(f"\nNote: no data for {missing_weather} — model used imputed medians. "
              f"Supply real weather for a sharper estimate.")
    print("=" * 66)


if __name__ == "__main__":
    main()
