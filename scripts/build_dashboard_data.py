"""
Precompute aggregates from California_Fire_Incidents.csv into a single JSON
file that the offline dashboard (dashboard/index.html) embeds directly.
Run: python scripts/build_dashboard_data.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from model_common import clean_lat_lon

ROOT = Path(__file__).parent.parent
CSV_PATH = ROOT / "California_Fire_Incidents.csv"
OUT_PATH = ROOT / "dashboard" / "data.json"
TEMPLATE_PATH = ROOT / "dashboard" / "template.html"
HTML_OUT_PATH = ROOT / "dashboard" / "index.html"
LARGE_FIRE_THRESHOLD_ACRES = 1000

SEASON_MAP = {12: "Winter", 1: "Winter", 2: "Winter",
              3: "Spring", 4: "Spring", 5: "Spring",
              6: "Summer", 7: "Summer", 8: "Summer",
              9: "Fall", 10: "Fall", 11: "Fall"}


def main():
    df = pd.read_csv(CSV_PATH, low_memory=False)
    df["Started"] = pd.to_datetime(df["Started"], errors="coerce", utc=True)
    cutoff = pd.Timestamp("1990-01-01", tz="UTC")
    df.loc[df["Started"] < cutoff, "Started"] = pd.NaT
    df = clean_lat_lon(df)
    df = df.dropna(subset=["Started", "AcresBurned", "Latitude", "Longitude"])

    df["Year"] = df["Started"].dt.year
    df["Month"] = df["Started"].dt.month
    df["Season"] = df["Month"].map(SEASON_MAP)
    df["CountyPrimary"] = df["Counties"].astype(str).str.split(",").str[0].str.strip()
    df["Large"] = (df["AcresBurned"] >= LARGE_FIRE_THRESHOLD_ACRES).astype(int)

    yearly = (df.groupby("Year")
                .agg(count=("UniqueId", "count"),
                     acres=("AcresBurned", "sum"),
                     large=("Large", "sum"))
                .reset_index()
                .sort_values("Year"))

    seasonal = (df.groupby("Season")
                  .agg(count=("UniqueId", "count"),
                       large=("Large", "sum"),
                       acres=("AcresBurned", "sum"))
                  .reindex(["Winter", "Spring", "Summer", "Fall"])
                  .reset_index())

    county = (df.groupby("CountyPrimary")
                .agg(count=("UniqueId", "count"),
                     acres=("AcresBurned", "sum"))
                .reset_index()
                .sort_values("acres", ascending=False)
                .head(15))

    points = df[["Latitude", "Longitude", "AcresBurned", "Year", "Large", "CountyPrimary", "Name"]].copy()
    points["AcresBurned"] = points["AcresBurned"].round(0)
    points = points.rename(columns={"Latitude": "lat", "Longitude": "lon",
                                     "AcresBurned": "acres", "CountyPrimary": "county"})

    stats = {
        "total_incidents": int(len(df)),
        "total_acres": float(df["AcresBurned"].sum()),
        "total_fatalities": int(pd.to_numeric(df["Fatalities"], errors="coerce").fillna(0).sum()),
        "total_structures_destroyed": int(pd.to_numeric(df["StructuresDestroyed"], errors="coerce").fillna(0).sum()),
        "date_min": df["Started"].min().strftime("%Y-%m-%d"),
        "date_max": df["Started"].max().strftime("%Y-%m-%d"),
        "small_count": int((df["Large"] == 0).sum()),
        "large_count": int((df["Large"] == 1).sum()),
    }

    payload = {
        "stats": stats,
        "yearly": yearly.to_dict(orient="records"),
        "seasonal": seasonal.to_dict(orient="records"),
        "county": county.to_dict(orient="records"),
        "points": points.to_dict(orient="records"),
    }

    OUT_PATH.parent.mkdir(exist_ok=True)
    data_json = json.dumps(payload, allow_nan=False)
    OUT_PATH.write_text(data_json)
    print(f"Wrote {OUT_PATH} ({OUT_PATH.stat().st_size / 1024:.0f} KB)")

    html = TEMPLATE_PATH.read_text().replace("__DASHBOARD_DATA_JSON__", data_json)
    HTML_OUT_PATH.write_text(html)
    print(f"Wrote {HTML_OUT_PATH} ({HTML_OUT_PATH.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
