"""
CALFIRE multi-threshold pattern analysis.

Runs the same leak-free feature pipeline and model bake-off as train_model.py
at several acreage cutoffs (10 / 50 / 100 / 500 / 1,000 acres) to see which
definitions of "large fire" the data actually carries a learnable signal for.
Each threshold gets its own county-historical-rate feature (the rate of
>=100-acre fires in a county is a different number than the rate of >=10-acre
fires), its own model bake-off, and its own holdout evaluation.

Saves:
  - output/threshold_analysis_report.md — comparison table across thresholds
  - output/multi_threshold_models.pkl   — one fitted pipeline per threshold,
    used by predict.py to print a full risk profile instead of one cutoff

Usage: python scripts/threshold_analysis.py
"""
import json
import logging
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_curve,
)

from model_common import (
    build_feature_lists, run_bakeoff, load_raw_data, add_threshold_target,
    RANDOM_STATE, SMOOTHING_ALPHA,
)

warnings.filterwarnings("ignore")
np.random.seed(RANDOM_STATE)

THRESHOLDS_ACRES = [10, 50, 100, 500, 1000]
TEST_SIZE = 0.2

ROOT = Path(__file__).parent.parent
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_PATH = OUTPUT_DIR / "multi_threshold_models.pkl"
REPORT_PATH = OUTPUT_DIR / "threshold_analysis_report.md"
DASHBOARD_JSON_PATH = ROOT / "dashboard" / "threshold_summary.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")
logger = logging.getLogger("CALFIRE-thresholds")


def run_for_threshold(base_df, threshold):
    df = add_threshold_target(base_df, threshold)
    numeric, categorical, geo = build_feature_lists(df)
    cols = numeric + categorical + geo + ["Large"]
    fe = df[cols].dropna(subset=["Large"])
    X = fe.drop(columns=["Large"])
    y = fe["Large"]

    large_pct = y.mean()
    logger.info("--- Threshold >= %d acres: %d/%d (%.1f%%) large ---", threshold, y.sum(), len(y), large_pct * 100)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    cv_results, best_name, pipe = run_bakeoff(X_train, y_train, numeric, categorical, geo)
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

    precisions, recalls, thr = precision_recall_curve(y_test, y_proba)
    f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    opt_idx = np.argmax(f1s[:-1])
    opt_thr = thr[opt_idx] if len(thr) else 0.5

    global_rate = df["Large"].mean()
    county_stats = df.groupby("CountyPrimary").agg(total=("Large", "count"), large=("Large", "sum"))
    county_rate_lookup = (
        (county_stats["large"] + SMOOTHING_ALPHA * global_rate) / (county_stats["total"] + SMOOTHING_ALPHA)
    ).to_dict()

    return {
        "threshold": threshold,
        "large_pct": large_pct,
        "n_large": int(y.sum()),
        "n_total": len(y),
        "cv_results": cv_results,
        "best_name": best_name,
        "metrics": metrics,
        "opt_thr": float(opt_thr),
        "pipeline": pipe,
        "numeric": numeric,
        "categorical": categorical,
        "geo": geo,
        "county_rate_lookup": county_rate_lookup,
        "global_rate": global_rate,
    }


def main():
    base_df = load_raw_data()
    global_county_stats = base_df.groupby("CountyPrimary").agg(
        lat=("Latitude", "mean"), lon=("Longitude", "mean"),
    )
    county_centroid_lookup = global_county_stats[["lat", "lon"]].apply(tuple, axis=1).to_dict()

    results = {t: run_for_threshold(base_df, t) for t in THRESHOLDS_ACRES}

    bundle = {
        "thresholds": {
            t: {
                "pipeline": r["pipeline"], "threshold_cutoff": r["opt_thr"],
                "numeric": r["numeric"], "categorical": r["categorical"], "geo": r["geo"],
                "model_name": r["best_name"],
                "county_rate_lookup": r["county_rate_lookup"], "global_rate": r["global_rate"],
                "test_roc_auc": r["metrics"]["ROC AUC"],
            }
            for t, r in results.items()
        },
        "county_centroid_lookup": county_centroid_lookup,
    }
    joblib.dump(bundle, MODEL_PATH)
    logger.info("Saved multi-threshold model bundle -> %s", MODEL_PATH)

    report = [
        "# CALFIRE Multi-Threshold Pattern Analysis\n\n",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n",
        "Same leak-free feature pipeline and model bake-off, run separately at "
        "each acreage cutoff, to see where the data actually carries a "
        "learnable signal for \"this fire will get big.\"\n\n",
        "| Threshold (ac) | % large | Best model | CV ROC-AUC | Test ROC-AUC | Test F1 |\n",
        "|---|---|---|---|---|---|\n",
    ]
    for t in THRESHOLDS_ACRES:
        r = results[t]
        cv_mean = r["cv_results"][r["best_name"]][0]
        report.append(
            f"| {t:,} | {r['large_pct']*100:.1f}% ({r['n_large']}/{r['n_total']}) | "
            f"{r['best_name']} | {cv_mean:.3f} | {r['metrics']['ROC AUC']:.3f} | "
            f"{r['metrics']['F1 Score']:.3f} |\n"
        )

    report.append("\n## Detail per threshold\n")
    for t in THRESHOLDS_ACRES:
        r = results[t]
        report.append(f"\n### >= {t:,} acres\n")
        report.append("Model bake-off (CV ROC-AUC):\n")
        for name, (mean, std) in sorted(r["cv_results"].items(), key=lambda kv: -kv[1][0]):
            report.append(f"- {name}: {mean:.4f} ± {std*2:.4f}\n")
        report.append(f"\nHoldout metrics ({r['best_name']}):\n")
        for k, v in r["metrics"].items():
            report.append(f"- {k}: {v:.4f}\n")

    REPORT_PATH.write_text("".join(report))
    logger.info("Report saved -> %s", REPORT_PATH)

    DASHBOARD_JSON_PATH.parent.mkdir(exist_ok=True)
    dashboard_summary = [
        {
            "threshold": t, "large_pct": results[t]["large_pct"],
            "best_model": results[t]["best_name"],
            "cv_roc_auc": results[t]["cv_results"][results[t]["best_name"]][0],
            "test_roc_auc": results[t]["metrics"]["ROC AUC"],
            "test_f1": results[t]["metrics"]["F1 Score"],
        }
        for t in THRESHOLDS_ACRES
    ]
    DASHBOARD_JSON_PATH.write_text(json.dumps(dashboard_summary))
    logger.info("Dashboard summary saved -> %s", DASHBOARD_JSON_PATH)

    best_t = max(results, key=lambda t: results[t]["metrics"]["ROC AUC"])
    logger.info("Most learnable threshold: >=%d acres (test ROC-AUC=%.3f)",
                best_t, results[best_t]["metrics"]["ROC AUC"])


if __name__ == "__main__":
    main()
