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
FORECAST_JSON_PATH = ROOT / "dashboard" / "forecast.json"
THRESHOLD_JSON_PATH = ROOT / "dashboard" / "threshold_summary.json"
HISTORICAL_JSON_PATH = ROOT / "dashboard" / "historical.json"
LARGE_FIRE_THRESHOLD_ACRES = 1000


def load_optional_json(path):
    if path.exists():
        return json.loads(path.read_text())
    print(f"Note: {path.name} not found — run its generating script to populate that dashboard panel.")
    return None

SEASON_MAP = {12: "Winter", 1: "Winter", 2: "Winter",
              3: "Spring", 4: "Spring", 5: "Spring",
              6: "Summer", 7: "Summer", 8: "Summer",
              9: "Fall", 10: "Fall", 11: "Fall"}
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
HEATMAP_TOP_N_COUNTIES = 15


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

    monthly = (df.groupby("Month")
                 .agg(count=("UniqueId", "count"), acres=("AcresBurned", "sum"))
                 .reindex(range(1, 13), fill_value=0)
                 .reset_index())
    monthly["MonthName"] = monthly["Month"].map(lambda m: MONTH_NAMES[m - 1])

    heatmap_counties = (df.groupby("CountyPrimary")["AcresBurned"].sum()
                          .sort_values(ascending=False).head(HEATMAP_TOP_N_COUNTIES).index.tolist())
    heat = (df[df["CountyPrimary"].isin(heatmap_counties)]
              .groupby(["CountyPrimary", "Month"])
              .agg(count=("UniqueId", "count"), acres=("AcresBurned", "sum"))
              .reindex(pd.MultiIndex.from_product([heatmap_counties, range(1, 13)],
                                                    names=["CountyPrimary", "Month"]), fill_value=0)
              .reset_index())
    heat["MonthName"] = heat["Month"].map(lambda m: MONTH_NAMES[m - 1])

    peak_month = (heat.loc[heat.groupby("CountyPrimary")["count"].idxmax()]
                       [["CountyPrimary", "MonthName", "count"]]
                       .rename(columns={"MonthName": "PeakMonth", "count": "PeakCount"}))
    county_totals = df[df["CountyPrimary"].isin(heatmap_counties)].groupby("CountyPrimary")["UniqueId"].count()
    peak_month = peak_month.merge(county_totals.rename("TotalCount"), on="CountyPrimary")
    peak_month["PeakShare"] = peak_month["PeakCount"] / peak_month["TotalCount"]
    peak_month = peak_month.set_index("CountyPrimary").loc[heatmap_counties].reset_index()

    # Structures/fatalities fields are only populated for a minority of fires
    # (~10% for structures, ~1% for fatalities) — presumably only reported
    # when nonzero/significant, not a reliable "0 means none" signal. So these
    # rankings are restricted to fires that actually have a reported value,
    # not the full dataset.
    df["StructuresDestroyedNum"] = pd.to_numeric(df["StructuresDestroyed"], errors="coerce")
    df["FatalitiesNum"] = pd.to_numeric(df["Fatalities"], errors="coerce")

    destructive = df[(df["StructuresDestroyedNum"] > 0) & (df["AcresBurned"] > 0)].copy()
    destructive["StructuresPer1000Acres"] = destructive["StructuresDestroyedNum"] / (destructive["AcresBurned"] / 1000)
    destructive = (destructive.sort_values("StructuresPer1000Acres", ascending=False)
                              .head(15)
                              [["Name", "CountyPrimary", "Year", "AcresBurned", "StructuresDestroyedNum", "StructuresPer1000Acres"]])

    deadly = (df[df["FatalitiesNum"] > 0]
                .sort_values("FatalitiesNum", ascending=False)
                .head(15)
                [["Name", "CountyPrimary", "Year", "AcresBurned", "FatalitiesNum"]])

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
        "monthly": monthly.to_dict(orient="records"),
        "heatmap": heat.to_dict(orient="records"),
        "peak_month": peak_month.to_dict(orient="records"),
        "destructive": destructive.to_dict(orient="records"),
        "deadly": deadly.to_dict(orient="records"),
        "reporting_coverage": {
            "structures_pct": float(df["StructuresDestroyedNum"].notna().mean()),
            "fatalities_pct": float(df["FatalitiesNum"].notna().mean()),
        },
        "points": points.to_dict(orient="records"),
        "forecast": load_optional_json(FORECAST_JSON_PATH),
        "thresholds": load_optional_json(THRESHOLD_JSON_PATH),
        "historical": load_optional_json(HISTORICAL_JSON_PATH),
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
