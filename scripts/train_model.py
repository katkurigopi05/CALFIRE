"""
CALFIRE large-fire prediction, v2 (flagship: >=1,000 acres).

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

For the same analysis repeated across multiple acreage thresholds (10/50/100/
500/1,000 ac), see scripts/threshold_analysis.py.

Usage: python scripts/train_model.py
"""
import logging
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, precision_recall_curve,
)

from model_common import (
    build_feature_lists, build_pipeline, run_bakeoff, load_raw_data, add_threshold_target,
    RANDOM_STATE, SMOOTHING_ALPHA,
)

warnings.filterwarnings("ignore")
np.random.seed(RANDOM_STATE)

LARGE_FIRE_THRESHOLD_ACRES = 1000
TEST_SIZE = 0.2

ROOT = Path(__file__).parent.parent
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_PATH = OUTPUT_DIR / "model_v2.pkl"
REPORT_PATH = OUTPUT_DIR / "model_v2_report.md"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")
logger = logging.getLogger("CALFIRE-v2")


def main():
    df = add_threshold_target(load_raw_data(), LARGE_FIRE_THRESHOLD_ACRES)
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

    cv_results, best_name, pipe = run_bakeoff(X_train, y_train, numeric, categorical, geo)
    logger.info("Best model: %s", best_name)
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
