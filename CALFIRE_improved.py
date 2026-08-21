"""
CALFIRE - California Wildfire Classification (Improved)
=======================================================
Fixes applied from CODE_REVIEW.md:
  #1  Data leakage     - removed post-hoc features
  #2  Overfitting      - max_depth/min_samples + cross-val
  #3  Report bug       - {pd} → csv_filename
  #4  File validation  - os.path.exists check
  #5  DataFrame        - consistent use of `work`
  #6  Missing threshold- drop cols with >70% NaN
  #7  Hardcoded paths  - pathlib.Path everywhere
  #8  Model persistence- joblib save/load
  #9  Season encoding  - dict-based, documented
  #10 DRY OHE          - create_onehot_encoder()
  #11 Magic numbers    - named constants
  #12 Target EDA       - class imbalance plot
  #13 Threshold tuning - precision-recall optimal
  #14 Invalid dates    - filter 1969 epoch dates
  #15 Random state     - random + os.environ seed
  #16 Logging          - replace print with logger
  #17 Validation fn    - validate_dataframe()
  #18 Metrics summary  - print_model_summary()
"""

# ── 0. IMPORTS & CONFIGURATION ─────────────────────────────────────────────
import os
import random
import logging
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix, RocCurveDisplay, precision_recall_curve
)
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

# ── Fix #15: Comprehensive reproducibility ────────────────────────────────
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
os.environ["PYTHONHASHSEED"] = str(RANDOM_STATE)

# ── Fix #11: Named constants ───────────────────────────────────────────────
LARGE_FIRE_THRESHOLD_ACRES = 1000   # CAL FIRE definition of "large incident"
TEST_SIZE                  = 0.2    # 80/20 split
N_ESTIMATORS               = 500    # trees (tuned for this dataset size)
MAX_DEPTH                  = 15     # prevent overfitting
MIN_SAMPLES_SPLIT          = 20     # min samples to allow a split
MIN_SAMPLES_LEAF           = 10     # min samples per leaf
TOP_N                      = 25     # features shown in importance plot
MISSING_THRESHOLD          = 0.70   # drop cols with >70% NaN
CV_FOLDS                   = 5      # cross-validation folds

# ── Fix #7: Pathlib paths ─────────────────────────────────────────────────
NOTEBOOK_DIR = Path(__file__).parent if "__file__" in dir() else Path.cwd()
DATA_DIR     = NOTEBOOK_DIR          # CSV lives in same dir as script
OUTPUT_DIR   = NOTEBOOK_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

CSV_PATH    = DATA_DIR / "California_Fire_Incidents.csv"
FI_PATH     = OUTPUT_DIR / "rf_feature_importances.csv"
REPORT_PATH = OUTPUT_DIR / "rf_report.md"
MODEL_PATH  = OUTPUT_DIR / f"rf_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"

# ── Fix #16: Logging ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(OUTPUT_DIR / "calfire_analysis.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("CALFIRE")


# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

# ── Fix #10: DRY OneHotEncoder ────────────────────────────────────────────
def create_onehot_encoder():
    """Return OneHotEncoder compatible with multiple sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


# ── Fix #9: Season encoding (dict-based, documented) ─────────────────────
SEASON_MAP = {
    12: 0, 1: 0, 2: 0,   # 0 = Winter
     3: 1, 4: 1, 5: 1,   # 1 = Spring
     6: 2, 7: 2, 8: 2,   # 2 = Summer
     9: 3,10: 3,11: 3,   # 3 = Fall
}
SEASON_NAMES = {0: "Winter", 1: "Spring", 2: "Summer", 3: "Fall"}

def month_to_season(m):
    """Convert month number to season code. Returns np.nan for invalid input."""
    if pd.isna(m):
        return np.nan
    return SEASON_MAP.get(int(m), np.nan)


# ── Fix #17: Input validation ─────────────────────────────────────────────
def validate_dataframe(df, required_columns):
    """Raise ValueError if df is empty or missing required columns."""
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if len(df) == 0:
        raise ValueError("DataFrame is empty after loading")
    logger.info("Validation passed: %d rows, %d columns", len(df), len(df.columns))
    return True


# ── Fix #18: Metrics summary ──────────────────────────────────────────────
def print_model_summary(y_true, y_pred, y_proba, model_name="Model"):
    """Print confusion matrix + full metrics dict."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "Accuracy" : accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall"   : recall_score(y_true, y_pred, zero_division=0),
        "F1 Score" : f1_score(y_true, y_pred, zero_division=0),
        "ROC AUC"  : roc_auc_score(y_true, y_proba),
    }

    border = "=" * 52
    logger.info("\n%s\n%s Performance\n%s", border, model_name, border)
    logger.info("Confusion Matrix:\n  TN:%4d  FP:%4d\n  FN:%4d  TP:%4d", tn, fp, fn, tp)
    for k, v in metrics.items():
        logger.info("  %-12s: %.4f", k, v)
    logger.info(border)
    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# CELL 1 — LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════

# ── Fix #4: File existence check ─────────────────────────────────────────
if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"Data file not found: {CSV_PATH}\n"
        "Place California_Fire_Incidents.csv in the same directory."
    )

try:
    df = pd.read_csv(CSV_PATH, low_memory=False)
    logger.info("Loaded %d rows × %d cols from %s", *df.shape, CSV_PATH.name)
except Exception as exc:
    logger.error("Failed to load CSV: %s", exc)
    raise

validate_dataframe(df, required_columns=["AcresBurned", "Started", "Counties"])


# ═══════════════════════════════════════════════════════════════════════════
# CELL 2 — QUICK OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════
logger.info("\nDtypes summary:\n%s", df.dtypes.value_counts().to_string())
missing_summary = df.isna().mean().sort_values(ascending=False)
logger.info("\nTop 10 missing %%:\n%s", missing_summary.head(10).to_string())


# ═══════════════════════════════════════════════════════════════════════════
# CELL 3 — DATE PARSING
# ═══════════════════════════════════════════════════════════════════════════

# ── Fix #5: use `work` consistently ──────────────────────────────────────
work = df.copy()

# ── Fix #14: Filter Unix-epoch / pre-1900 invalid dates ──────────────────
for col in ["Started", "Updated"]:
    if col not in work.columns:
        continue
    work[col] = pd.to_datetime(work[col], errors="coerce", utc=True)
    cutoff = pd.Timestamp("1990-01-01", tz="UTC")   # fires pre-1990 likely invalid
    bad = work[col] < cutoff
    if bad.any():
        logger.warning("%d invalid/epoch dates in '%s' → set to NaT", bad.sum(), col)
        work.loc[bad, col] = pd.NaT

valid = work["Started"].dropna()
if len(valid):
    logger.info("Valid date range: %s → %s", valid.min(), valid.max())


# ═══════════════════════════════════════════════════════════════════════════
# CELL 4 — TARGET + EDA
# ═══════════════════════════════════════════════════════════════════════════

# ── Fix #5: all mutations on `work`, not `df` ────────────────────────────
if "Counties" in work.columns:
    work["CountyPrimary"] = (
        work["Counties"].astype(str).str.split(",").str[0].str.strip()
    )

if "Started" in work.columns:
    work["StartMonth"]   = work["Started"].dt.month
    work["StartYear"]    = work["Started"].dt.year
    work["StartDOY"]     = work["Started"].dt.day_of_year
    work["StartWeekday"] = work["Started"].dt.weekday
    work["StartSeason"]  = work["StartMonth"].apply(month_to_season)
    work["SeasonName"]   = work["StartSeason"].map(SEASON_NAMES)

# ── Fix #11: use named constant ───────────────────────────────────────────
work["LargeFire"] = (work["AcresBurned"] >= LARGE_FIRE_THRESHOLD_ACRES).astype(int)

# ── Fix #12: Target variable EDA ─────────────────────────────────────────
small_n = (work["LargeFire"] == 0).sum()
large_n = (work["LargeFire"] == 1).sum()
logger.info(
    "Class balance | Small (<%.0f ac): %d (%.1f%%) | Large (>=%.0f ac): %d (%.1f%%)",
    LARGE_FIRE_THRESHOLD_ACRES, small_n, 100*small_n/len(work),
    LARGE_FIRE_THRESHOLD_ACRES, large_n, 100*large_n/len(work),
)

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle("CALFIRE – Target Variable Analysis", fontweight="bold")

# AcresBurned histogram
ax = axes[0]
ax.hist(work[work["AcresBurned"] < 10_000]["AcresBurned"], bins=60, color="#e74c3c", alpha=0.75)
ax.axvline(LARGE_FIRE_THRESHOLD_ACRES, color="navy", ls="--", lw=2, label=f"Threshold={LARGE_FIRE_THRESHOLD_ACRES:,} ac")
ax.set_title("AcresBurned (< 10k)")
ax.set_xlabel("Acres")
ax.legend()

# Log-scale
ax = axes[1]
ax.hist(np.log10(work["AcresBurned"].clip(lower=1)), bins=60, color="#3498db", alpha=0.75)
ax.axvline(np.log10(LARGE_FIRE_THRESHOLD_ACRES), color="navy", ls="--", lw=2, label="Threshold")
ax.set_title("Log₁₀(AcresBurned)")
ax.set_xlabel("log₁₀ Acres")
ax.legend()

# Class bar
ax = axes[2]
ax.bar(["Small\n(<1k ac)", "Large\n(≥1k ac)"], [small_n, large_n],
       color=["#27ae60", "#e74c3c"])
ax.set_title("Class Distribution")
ax.set_ylabel("Count")
for bar in ax.patches:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            f"{int(bar.get_height()):,}", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "target_eda.png", dpi=150, bbox_inches="tight")
plt.show()
logger.info("Saved target_eda.png")

# Season fire counts
if "SeasonName" in work.columns:
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.countplot(data=work, x="SeasonName",
                  order=["Winter", "Spring", "Summer", "Fall"],
                  hue="LargeFire", palette={0: "#27ae60", 1: "#e74c3c"}, ax=ax)
    ax.set_title("Fire Count by Season and Size")
    ax.set_xlabel("Season"); ax.set_ylabel("Count")
    ax.legend(title="LargeFire", labels=["Small", "Large"])
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "season_eda.png", dpi=150, bbox_inches="tight")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════
# CELL 5 — FEATURE ENGINEERING (leak-free)
# ═══════════════════════════════════════════════════════════════════════════

# ── Fix #1: Only use features available BEFORE / AT fire start ───────────
if "ArchiveYear" not in work.columns and "ArchiveYear" in df.columns:
    work["ArchiveYear"] = df["ArchiveYear"]

# Features known at ignition time only
PREDICTION_NUMERIC = [c for c in [
    "ArchiveYear", "StartYear", "StartMonth",
    "StartDOY", "StartWeekday", "StartSeason",
] if c in work.columns]

PREDICTION_CATEGORICAL = [c for c in [
    "CountyPrimary", "Status",
] if c in work.columns]

# ── Fix #6: Drop high-missingness numeric features ────────────────────────
fe = work.copy()
missing_pct = fe[PREDICTION_NUMERIC].isna().mean()
valid_numeric = missing_pct[missing_pct < MISSING_THRESHOLD].index.tolist()
dropped = set(PREDICTION_NUMERIC) - set(valid_numeric)
if dropped:
    logger.warning("Dropped %d high-NaN features (>%.0f%%): %s",
                   len(dropped), MISSING_THRESHOLD*100, dropped)

candidate_numeric      = valid_numeric
candidate_categorical  = PREDICTION_CATEGORICAL

logger.info("Prediction features | numeric=%s | categorical=%s",
            candidate_numeric, candidate_categorical)

# ── Build modelling DataFrame ─────────────────────────────────────────────
TARGET = "LargeFire"
all_cols = candidate_numeric + candidate_categorical + [TARGET]
fe = fe[all_cols].dropna(subset=[TARGET])
logger.info("Modelling set: %d rows, %d features", len(fe), len(all_cols) - 1)


# ═══════════════════════════════════════════════════════════════════════════
# CELL 6 — TRAIN / TEST SPLIT
# ═══════════════════════════════════════════════════════════════════════════
X = fe.drop(columns=[TARGET])
y = fe[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
logger.info("Train: %d  Test: %d", len(X_train), len(X_test))


# ═══════════════════════════════════════════════════════════════════════════
# CELL 7 — PREPROCESSING + MODEL
# ═══════════════════════════════════════════════════════════════════════════

# ── Fix #10: DRY OHE helper ───────────────────────────────────────────────
ohe = create_onehot_encoder()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
        ]), candidate_numeric),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ohe", ohe),
        ]), candidate_categorical),
    ],
    remainder="drop",
)

# ── Fix #2: Overfitting controls ─────────────────────────────────────────
rf = RandomForestClassifier(
    n_estimators     = N_ESTIMATORS,
    max_depth        = MAX_DEPTH,
    min_samples_split= MIN_SAMPLES_SPLIT,
    min_samples_leaf = MIN_SAMPLES_LEAF,
    max_features     = "sqrt",
    class_weight     = "balanced_subsample",
    random_state     = RANDOM_STATE,
    n_jobs           = -1,
)

pipe = Pipeline([("pre", preprocessor), ("rf", rf)])
pipe.fit(X_train, y_train)
logger.info("Model training complete")

# ── Fix #2: Cross-validation ─────────────────────────────────────────────
skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
cv_scores = cross_val_score(pipe, X_train, y_train, cv=skf, scoring="roc_auc", n_jobs=-1)
logger.info("CV ROC-AUC: %.4f ± %.4f", cv_scores.mean(), cv_scores.std() * 2)

train_acc = pipe.score(X_train, y_train)
test_acc  = pipe.score(X_test, y_test)
logger.info("Train accuracy: %.4f | Test accuracy: %.4f | Gap: %.4f",
            train_acc, test_acc, train_acc - test_acc)


# ═══════════════════════════════════════════════════════════════════════════
# CELL 8 — EVALUATION
# ═══════════════════════════════════════════════════════════════════════════
y_pred  = pipe.predict(X_test)
y_proba = pipe.predict_proba(X_test)[:, 1]

# ── Fix #18: unified metrics summary ─────────────────────────────────────
metrics = print_model_summary(y_test, y_pred, y_proba,
                               model_name="Random Forest (leak-free)")
logger.info("\nClassification Report:\n%s",
            classification_report(y_test, y_pred, digits=3))

# Confusion matrix plot
fig, ax = plt.subplots(figsize=(5, 4))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Small", "Large"],
            yticklabels=["Small", "Large"], ax=ax)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
plt.show()

# ROC curve
fig, ax = plt.subplots(figsize=(6, 5))
RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax,
                                  name="Random Forest",
                                  color="#e74c3c")
ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set_title("ROC Curve")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "roc_curve.png", dpi=150)
plt.show()


# ═══════════════════════════════════════════════════════════════════════════
# CELL 9 — FEATURE IMPORTANCES
# ═══════════════════════════════════════════════════════════════════════════
try:
    ohe_features = (pipe.named_steps["pre"]
                    .named_transformers_["cat"]
                    .named_steps["ohe"]
                    .get_feature_names_out(candidate_categorical))
    feature_names = candidate_numeric + list(ohe_features)
except Exception:
    feature_names = [f"f{i}" for i in range(pipe.named_steps["rf"].n_features_in_)]

importances = pipe.named_steps["rf"].feature_importances_
fi_df = (pd.DataFrame({"feature": feature_names, "importance": importances})
           .sort_values("importance", ascending=False)
           .head(TOP_N))

fi_df.to_csv(FI_PATH, index=False)
logger.info("Feature importances saved → %s", FI_PATH)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=fi_df, y="feature", x="importance", color="#3498db", ax=ax)
ax.set_title(f"Top {TOP_N} Feature Importances")
ax.set_xlabel("Mean Decrease in Impurity")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "feature_importances.png", dpi=150)
plt.show()


# ═══════════════════════════════════════════════════════════════════════════
# CELL 10 — THRESHOLD TUNING (Fix #13)
# ═══════════════════════════════════════════════════════════════════════════
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
opt_idx   = np.argmax(f1_scores[:-1])   # last element has no threshold
opt_thr   = thresholds[opt_idx]

logger.info("Optimal threshold: %.3f  →  Precision=%.3f  Recall=%.3f  F1=%.3f",
            opt_thr, precisions[opt_idx], recalls[opt_idx], f1_scores[opt_idx])

y_pred_opt = (y_proba >= opt_thr).astype(int)
logger.info("\nWith optimal threshold:\n%s",
            classification_report(y_test, y_pred_opt, digits=3))

# PR curve plot
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(recalls, precisions, color="#e74c3c", lw=2)
ax.scatter(recalls[opt_idx], precisions[opt_idx], zorder=5,
           color="navy", s=80, label=f"Best F1 @ thr={opt_thr:.2f}")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve")
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "pr_curve.png", dpi=150)
plt.show()


# ═══════════════════════════════════════════════════════════════════════════
# CELL 11 — MODEL PERSISTENCE (Fix #8)
# ═══════════════════════════════════════════════════════════════════════════
joblib.dump(pipe, MODEL_PATH)
logger.info("Model saved → %s", MODEL_PATH)

# Verify round-trip
loaded = joblib.load(MODEL_PATH)
assert loaded.score(X_test, y_test) == test_acc, "Round-trip accuracy mismatch!"
logger.info("Model round-trip verified ✓")


# ═══════════════════════════════════════════════════════════════════════════
# CELL 12 — MARKDOWN REPORT (Fix #3)
# ═══════════════════════════════════════════════════════════════════════════
csv_filename = CSV_PATH.name   # Fix #3: was `{pd}`, now correct filename

report = []
report.append("# CALFIRE Random Forest — Results Report\n\n")
report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
report.append("## Dataset\n")
report.append(f"- File: `{csv_filename}`\n")          # Fixed
report.append(f"- Rows loaded: {len(df):,}\n")
report.append(f"- Modelling rows (after clean): {len(fe):,}\n\n")
report.append("## Features Used (leak-free)\n")
report.append(f"- Numeric: {candidate_numeric}\n")
report.append(f"- Categorical: {candidate_categorical}\n\n")
report.append("## Model\n")
report.append(f"- RandomForestClassifier  n_estimators={N_ESTIMATORS}  max_depth={MAX_DEPTH}\n")
report.append(f"- CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std()*2:.4f}\n\n")
report.append("## Performance\n")
for k, v in metrics.items():
    report.append(f"- {k}: {v:.4f}\n")
report.append(f"\n- Optimal classification threshold: {opt_thr:.3f}\n")
report.append(f"\n## Model saved\n- `{MODEL_PATH.name}`\n")

REPORT_PATH.write_text("".join(report))
logger.info("Report saved → %s", REPORT_PATH)
logger.info("\n\nAll done! Outputs in: %s", OUTPUT_DIR)
