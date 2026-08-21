"""Shared pieces between train_model.py, threshold_analysis.py, and predict.py.

GeoCluster must live in its own importable module (not inside a __main__
script) so joblib can unpickle it later from a different entry point.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

RANDOM_STATE = 42
N_GEO_CLUSTERS = 8
SMOOTHING_ALPHA = 10  # Laplace smoothing strength for county historical rate
CV_FOLDS = 5

ROOT = Path(__file__).parent.parent
CSV_PATH = ROOT / "California_Fire_Incidents.csv"
WEATHER_PATH = ROOT / "weather_data.csv"
HISTORIC_CSV_PATH = ROOT / "California_Historic_Fire_Perimeters.csv"
UNITS_CSV_PATH = ROOT / "CALFIRE_Administrative_Units.csv"

logger = logging.getLogger("CALFIRE")

# CAL FIRE FRAP standard cause codes (per the Wildland Fire Perimeters
# metadata). Best-effort mapping — verify against the source metadata doc if
# exact code semantics matter for a specific analysis.
CAUSE_CODE_MAP = {
    1: "Lightning", 2: "Equipment Use", 3: "Smoking", 4: "Campfire",
    5: "Debris", 6: "Railroad", 7: "Arson", 8: "Playing with Fire",
    9: "Miscellaneous", 10: "Vehicle", 11: "Power Line", 12: "Firefighter Training",
    13: "Non-Firefighter Training", 14: "Unknown/Unidentified", 15: "Structure",
    16: "Aircraft", 17: "Volcanic", 18: "Escaped Prescribed Burn",
    19: "Illegal Alien Campfire",
}


class GeoCluster(BaseEstimator, TransformerMixin):
    """Assigns each (lat, lon) to a KMeans region cluster, fit on training data only."""

    def __init__(self, n_clusters=N_GEO_CLUSTERS, random_state=RANDOM_STATE):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit(self, X, y=None):
        self.kmeans_ = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        self.kmeans_.fit(X)
        return self

    def transform(self, X):
        labels = self.kmeans_.predict(np.asarray(X))
        return labels.reshape(-1, 1).astype(str)


# Generous CA-plus-border-states bounding box; anything outside this after
# the fixes below is treated as unrecoverable bad data, not a real location.
CA_LAT_RANGE = (30.0, 44.0)
CA_LON_RANGE = (-126.0, -112.0)


def clean_lat_lon(df, lat_col="Latitude", lon_col="Longitude"):
    """Fix known data-quality issues in the CAL FIRE lat/lon columns:
    - (0, 0) "null island" rows are missing data, not a real location
    - some rows have latitude/longitude swapped (lat > 90 is impossible)
    - a few swapped rows also lost the negative sign on longitude
    Anything that still falls outside a generous CA bounding box afterwards
    is set to NaN so downstream code drops or imputes it instead of silently
    treating garbage coordinates as a real fire location."""
    df = df.copy()
    swapped = (df[lat_col].abs() > 90) & (df[lon_col].abs() <= 90)
    df.loc[swapped, [lat_col, lon_col]] = df.loc[swapped, [lon_col, lat_col]].values

    flipped_sign = (df[lon_col] > 0) & (df[lon_col].between(-CA_LON_RANGE[1], -CA_LON_RANGE[0]))
    df.loc[flipped_sign, lon_col] = -df.loc[flipped_sign, lon_col]

    null_island = (df[lat_col] == 0) & (df[lon_col] == 0)
    out_of_range = (
        ~df[lat_col].between(*CA_LAT_RANGE) | ~df[lon_col].between(*CA_LON_RANGE)
    )
    df.loc[null_island | out_of_range, [lat_col, lon_col]] = np.nan
    return df


def add_county_history_rate(df, county_col, date_col, target_col, alpha=SMOOTHING_ALPHA):
    """Smoothed rate of `target_col` in `county_col`, using only rows strictly
    earlier in `date_col`. Safe to compute on the whole dataset because it
    only ever looks backward in time relative to each row."""
    df = df.sort_values(date_col).reset_index(drop=True)
    grp = df.groupby(county_col)[target_col]
    prior_sum = grp.cumsum() - df[target_col]
    prior_count = grp.cumcount()
    global_rate = df[target_col].expanding().mean().shift(1).fillna(df[target_col].mean())
    df["CountyHistoricalRate"] = (prior_sum + alpha * global_rate) / (prior_count + alpha)
    return df


def load_raw_data():
    """Load, clean, and feature-engineer everything that does NOT depend on
    a large-fire acreage threshold: dates, geography, weather. Threshold-
    dependent pieces (the target and the county historical rate) are added
    separately by add_threshold_target, since "large" means different things
    at 10 acres vs 1,000 acres."""
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH, low_memory=False)
    df["Started"] = pd.to_datetime(df["Started"], errors="coerce", utc=True)
    cutoff = pd.Timestamp("1990-01-01", tz="UTC")
    df.loc[df["Started"] < cutoff, "Started"] = pd.NaT
    df = clean_lat_lon(df)
    before = len(df)
    df = df.dropna(subset=["Started", "AcresBurned", "Latitude", "Longitude"])
    logger.info("Dropped %d rows with missing date/acres/lat-lon (of %d)", before - len(df), before)

    df["CountyPrimary"] = df["Counties"].astype(str).str.split(",").str[0].str.strip()
    df["StartMonth"] = df["Started"].dt.month
    df["StartYear"] = df["Started"].dt.year
    df["StartDOY"] = df["Started"].dt.dayofyear
    df["StartWeekday"] = df["Started"].dt.weekday
    df["DOY_sin"] = np.sin(2 * np.pi * df["StartDOY"] / 365.25)
    df["DOY_cos"] = np.cos(2 * np.pi * df["StartDOY"] / 365.25)
    df["DrySeason"] = df["StartMonth"].between(6, 10).astype(int)

    if WEATHER_PATH.exists():
        weather = pd.read_csv(WEATHER_PATH)
        before = len(df)
        df = df.merge(weather, on="UniqueId", how="left")
        matched = df["temp_max_c"].notna().sum() if "temp_max_c" in df.columns else 0
        logger.info("Merged weather_data.csv: %d/%d incidents matched", matched, before)
    else:
        logger.warning("weather_data.csv not found — training without real weather features. "
                        "Run scripts/fetch_weather.py somewhere with internet access to add it.")

    return df


def load_dates_and_acres():
    """Lightweight loader for time-series work: only needs a valid date and
    acreage, unlike load_raw_data() which also requires valid lat/lon (and so
    would needlessly drop ~160 fires that have no location but do have a
    perfectly good date/acres for a monthly aggregate)."""
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH, low_memory=False)
    df["Started"] = pd.to_datetime(df["Started"], errors="coerce", utc=True)
    cutoff = pd.Timestamp("1990-01-01", tz="UTC")
    df.loc[df["Started"] < cutoff, "Started"] = pd.NaT
    return df.dropna(subset=["Started", "AcresBurned"])


def load_historic_perimeters():
    """Load the 1878-2025 CAL FIRE FRAP fire perimeter dataset (no lat/lon —
    it's polygon perimeters, not incident points; see fetch instructions in
    README). Reliable Year/Acres for the whole span; Alarm Date is missing
    ~23% of the time (worse in early decades), Containment Date ~54%."""
    if not HISTORIC_CSV_PATH.exists():
        raise FileNotFoundError(
            f"{HISTORIC_CSV_PATH} not found. Download the 'California Fire "
            "Perimeters (all)' CSV from CAL FIRE FRAP's historical fire "
            "perimeter dataset and place it at the repo root."
        )
    df = pd.read_csv(HISTORIC_CSV_PATH, low_memory=False)
    df["AlarmDate"] = pd.to_datetime(df["Alarm Date"], errors="coerce", format="mixed")
    df["ContainDate"] = pd.to_datetime(df["Containment Date"], errors="coerce", format="mixed")
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Acres"] = pd.to_numeric(df["GIS Calculated Acres"], errors="coerce")
    df["CauseName"] = df["Cause"].map(CAUSE_CODE_MAP).fillna("Unknown/Unidentified")
    df = df.dropna(subset=["Year", "Acres"])
    df["Year"] = df["Year"].astype(int)
    return df


def add_calfire_unit_labels(df, unit_col="Unit ID"):
    """Label each fire with its CAL FIRE unit name + Northern/Southern region,
    using CALFIRE_Administrative_Units.csv. Only covers CAL FIRE's own
    jurisdiction (state responsibility areas) — roughly 46% of fires in the
    historical dataset are on federal land (USFS/NPS/BLM) with different unit
    codes this lookup doesn't have, so those are explicitly labeled
    "Federal/Other Agency" rather than silently dropped or mis-mapped."""
    df = df.copy()
    if not UNITS_CSV_PATH.exists():
        df["CalFireUnitName"] = "Unknown"
        df["CalFireRegion"] = "Unknown"
        return df
    units = pd.read_csv(UNITS_CSV_PATH)
    code_to_name = dict(zip(units["Unit Code"], units["Unit"]))
    code_to_region = dict(zip(units["Unit Code"], units["Region"]))
    df["CalFireUnitName"] = df[unit_col].map(code_to_name)
    df["CalFireRegion"] = df[unit_col].map(code_to_region).fillna("Federal/Other Agency")
    df.loc[df["CalFireUnitName"].isna(), "CalFireUnitName"] = "Federal/Other Agency"
    return df


def add_threshold_target(df, threshold_acres, alpha=SMOOTHING_ALPHA):
    """Return a copy of df with the binary target ('Large' = AcresBurned >=
    threshold_acres) and its matching leak-free county historical rate."""
    df = df.copy()
    df["Large"] = (df["AcresBurned"] >= threshold_acres).astype(int)
    df = add_county_history_rate(df, "CountyPrimary", "Started", "Large", alpha=alpha)
    return df


def build_feature_lists(df):
    numeric = [c for c in [
        "StartYear", "DOY_sin", "DOY_cos", "StartWeekday", "DrySeason",
        "CountyHistoricalRate",
        "temp_max_c", "temp_min_c", "precip_mm", "wind_max_kmh", "et0_mm", "humidity_mean_pct",
    ] if c in df.columns]
    categorical = [c for c in ["Status"] if c in df.columns]
    geo = ["Latitude", "Longitude"]
    return numeric, categorical, geo


def create_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_pipeline(numeric, categorical, geo, model):
    pre = ColumnTransformer(transformers=[
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ohe", create_ohe()),
        ]), categorical),
        ("geo", Pipeline([
            ("cluster", GeoCluster()),
            ("ohe", create_ohe()),
        ]), geo),
    ])
    return Pipeline([("pre", pre), ("model", model)])


def get_candidate_models(random_state=RANDOM_STATE):
    """Fresh (unfitted) model instances for the bake-off. A function, not a
    module-level dict, so repeated calls (one per threshold) never share
    fitted state."""
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=500, max_depth=15, min_samples_split=20, min_samples_leaf=10,
            max_features="sqrt", class_weight="balanced_subsample",
            random_state=random_state, n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_depth=6, learning_rate=0.05, max_iter=300,
            l2_regularization=1.0, random_state=random_state,
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=random_state,
        ),
    }


def run_bakeoff(X_train, y_train, numeric, categorical, geo, cv_folds=CV_FOLDS, random_state=RANDOM_STATE):
    """Cross-validate each candidate model, return (cv_results, best_name,
    unfitted best pipeline). cv_results maps name -> (mean, std) ROC-AUC."""
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    candidates = get_candidate_models(random_state)
    cv_results = {}
    for name, model in candidates.items():
        pipe = build_pipeline(numeric, categorical, geo, model)
        scores = cross_val_score(pipe, X_train, y_train, cv=skf, scoring="roc_auc", n_jobs=-1)
        cv_results[name] = (scores.mean(), scores.std())
        logger.info("%-22s CV ROC-AUC: %.4f ± %.4f", name, scores.mean(), scores.std() * 2)
    best_name = max(cv_results, key=lambda n: cv_results[n][0])
    best_pipe = build_pipeline(numeric, categorical, geo, get_candidate_models(random_state)[best_name])
    return cv_results, best_name, best_pipe
