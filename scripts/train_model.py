"""
CALFIRE large-fire prediction, v2.

Builds on CALFIRE_improved.py's leak-free design (only features known at
ignition time) and adds:
  - cyclical day-of-year encoding (sin/cos) instead of a raw linear day count
  - dry-season flag (Jun-Oct, CA fire season)
  - leak-free county fire-history rate: for each fire, the smoothed historical
    rate of large fires in that county using only fires strictly before it in
    time (a real quantity available at prediction time, computed once on the
    sorted timeline, not per train/test split)
  - a geographic cluster feature (KMeans on lat/lon, fit inside the pipeline
    on the training fold only) capturing regional fire behavior
  - optional real weather features (temp/precip/wind/ET0/humidity) merged in
    from weather_data.csv if present (see scripts/fetch_weather.py)
  - a model bake-off: RandomForest vs HistGradientBoosting vs LogisticRegression,
    picked by cross-validated ROC-AUC

Usage: python scripts/train_model.py
"""
import logging
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix, precision_recall_curve,
)

from model_common import GeoCluster, add_county_history_rate, clean_lat_lon, RANDOM_STATE, SMOOTHING_ALPHA

warnings.filterwarnings("ignore")

np.random.seed(RANDOM_STATE)

LARGE_FIRE_THRESHOLD_ACRES = 1000
TEST_SIZE = 0.2
CV_FOLDS = 5

ROOT = Path(__file__).parent.parent
CSV_PATH = ROOT / "California_Fire_Incidents.csv"
WEATHER_PATH = ROOT / "weather_data.csv"
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_PATH = OUTPUT_DIR / "model_v2.pkl"
REPORT_PATH = OUTPUT_DIR / "model_v2_report.md"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")
logger = logging.getLogger("CALFIRE-v2")


def load_and_prepare():
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
    df["Large"] = (df["AcresBurned"] >= LARGE_FIRE_THRESHOLD_ACRES).astype(int)

    df["StartMonth"] = df["Started"].dt.month
    df["StartYear"] = df["Started"].dt.year
    df["StartDOY"] = df["Started"].dt.dayofyear
    df["StartWeekday"] = df["Started"].dt.weekday
    df["DOY_sin"] = np.sin(2 * np.pi * df["StartDOY"] / 365.25)
    df["DOY_cos"] = np.cos(2 * np.pi * df["StartDOY"] / 365.25)
    df["DrySeason"] = df["StartMonth"].between(6, 10).astype(int)

    df = add_county_history_rate(df, "CountyPrimary", "Started", "Large")

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


def build_feature_lists(df):
    numeric = [c for c in [
        "StartYear", "DOY_sin", "DOY_cos", "StartWeekday", "DrySeason",
        "CountyHistoricalRate",
        "temp_max_c", "temp_min_c", "precip_mm", "wind_max_kmh", "et0_mm", "humidity_mean_pct",
    ] if c in df.columns]
    categorical = [c for c in ["Status"] if c in df.columns]
    geo = ["Latitude", "Longitude"]
    return numeric, categorical, geo


def build_pipeline(numeric, categorical, geo, model):
    def create_ohe():
        try:
            return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            return OneHotEncoder(handle_unknown="ignore", sparse=False)

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


def main():
    df = load_and_prepare()
    numeric, categorical, geo = build_feature_lists(df)
    logger.info("Numeric features: %s", numeric)
    logger.info("Categorical features: %s", categorical)

    cols = numeric + categorical + geo + ["Large"]
    fe = df[cols].dropna(subset=["Large"])
    X = fe.drop(columns=["Large"])
    y = fe["Large"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info("Train: %d  Test: %d", len(X_train), len(X_test))

    candidates = {
        "RandomForest": RandomForestClassifier(
            n_estimators=500, max_depth=15, min_samples_split=20, min_samples_leaf=10,
            max_features="sqrt", class_weight="balanced_subsample",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_depth=6, learning_rate=0.05, max_iter=300,
            l2_regularization=1.0, random_state=RANDOM_STATE,
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE,
        ),
    }

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_results = {}
    for name, model in candidates.items():
        pipe = build_pipeline(numeric, categorical, geo, model)
        scores = cross_val_score(pipe, X_train, y_train, cv=skf, scoring="roc_auc", n_jobs=-1)
        cv_results[name] = (scores.mean(), scores.std())
        logger.info("%-22s CV ROC-AUC: %.4f ± %.4f", name, scores.mean(), scores.std() * 2)

    best_name = max(cv_results, key=lambda n: cv_results[n][0])
    logger.info("Best model: %s", best_name)

    pipe = build_pipeline(numeric, categorical, geo, candidates[best_name])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC AUC": roc_auc_score(y_test, y_proba),
    }
    logger.info("Test metrics: %s", metrics)
    logger.info("\n%s", classification_report(y_test, y_pred, digits=3))

    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    opt_idx = np.argmax(f1s[:-1])
    opt_thr = thresholds[opt_idx]
    logger.info("Optimal threshold: %.3f (F1=%.3f)", opt_thr, f1s[opt_idx])

    # Lookups needed to score a brand-new (future) fire at inference time:
    # the county's full historical large-fire rate to date, and a location
    # fallback (county centroid) for when lat/lon isn't supplied.
    global_rate = df["Large"].mean()
    county_stats = df.groupby("CountyPrimary").agg(
        total=("Large", "count"), large=("Large", "sum"),
        lat=("Latitude", "mean"), lon=("Longitude", "mean"),
    )
    county_rate_lookup = (
        (county_stats["large"] + SMOOTHING_ALPHA * global_rate) / (county_stats["total"] + SMOOTHING_ALPHA)
    ).to_dict()
    county_centroid_lookup = county_stats[["lat", "lon"]].apply(tuple, axis=1).to_dict()

    joblib.dump({
        "pipeline": pipe, "threshold": opt_thr, "numeric": numeric,
        "categorical": categorical, "geo": geo, "model_name": best_name,
        "county_rate_lookup": county_rate_lookup, "global_rate": global_rate,
        "county_centroid_lookup": county_centroid_lookup,
        "has_weather": "temp_max_c" in df.columns,
    }, MODEL_PATH)
    logger.info("Model saved -> %s", MODEL_PATH)

    report = [
        "# CALFIRE Model v2 — Results\n\n",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n",
        f"Weather features: {'yes' if 'temp_max_c' in df.columns else 'no (weather_data.csv not found)'}\n\n",
        "## Model bake-off (5-fold CV ROC-AUC)\n",
    ]
    for name, (mean, std) in sorted(cv_results.items(), key=lambda kv: -kv[1][0]):
        report.append(f"- {name}: {mean:.4f} ± {std*2:.4f}\n")
    report.append(f"\n**Selected: {best_name}**\n\n")
    report.append("## Holdout test performance\n")
    for k, v in metrics.items():
        report.append(f"- {k}: {v:.4f}\n")
    report.append(f"\n- Optimal classification threshold: {opt_thr:.3f}\n")
    report.append(f"\n## Features\n- Numeric: {numeric}\n- Categorical: {categorical}\n- Geo cluster: {geo}\n")
    REPORT_PATH.write_text("".join(report))
    logger.info("Report saved -> %s", REPORT_PATH)


if __name__ == "__main__":
    main()
