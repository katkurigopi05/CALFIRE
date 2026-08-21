# Code Review: CALFIRE Random Forest Notebook

## Executive Summary
This document provides a comprehensive review of the CALFIRE(RF) RN7945.ipynb notebook, identifying all mistakes, issues, and areas for improvement with detailed solutions.

---

## Critical Issues

### 1. **DATA LEAKAGE - Most Critical Issue**

**Location**: Cell 5 (Feature Engineering) and Cell 7 (Model Training)

**Problem**:
```python
# These features are included in the model but are ONLY available AFTER the fire has occurred:
- DurationDays (calculated from Updated - Started timestamps)
- PercentContained (only known during/after firefighting)
- PersonnelInvolved (deployment happens during the fire)
- Engines, Dozers, AirTankers, Helicopters, CrewsInvolved, WaterTenders (all post-hoc)
```

**Why This Is Critical**:
- The model achieves 88.4% test accuracy and 99.9% training accuracy
- When these leakage features are removed (Cell 10), accuracy drops to 82.3% with much lower recall (19.7%)
- This means the model is **NOT** predicting whether a fire will be large, but rather **detecting** that it WAS large based on response data
- The model is essentially useless for practical prediction purposes

**Solution**:
```python
# In Cell 5, create TWO feature sets clearly:

# Features available BEFORE/AT FIRE START (prediction-ready):
prediction_features_numeric = [
    "ArchiveYear", "StartYear", "StartMonth", "StartDOY",
    "StartWeekday", "StartSeason"
]
prediction_features_categorical = ["CountyPrimary", "Status"]

# Features available ONLY AFTER (for analysis only):
post_hoc_features = [
    "DurationDays", "Engines", "Dozers", "AirTankers",
    "Helicopters", "CrewsInvolved", "PersonnelInvolved",
    "WaterTenders", "PercentContained"
]

# ALWAYS train models using prediction_features only
# Add clear warnings in comments about data leakage
```

---

### 2. **Overfitting - Very High Train vs Test Accuracy Gap**

**Location**: Cell 7 (Model Training)

**Problem**:
```python
# Current results show severe overfitting:
Train accuracy: 0.9992343032159265  # 99.92%
Test  accuracy: 0.8837920489296636  # 88.38%
```

**Why This Is a Problem**:
- 11.5% gap between train and test accuracy indicates the model memorized training data
- RandomForest with 500 trees and no depth limits can easily overfit
- This reduces generalization to new fire incidents

**Solution**:
```python
# In Cell 7, add hyperparameters to prevent overfitting:

rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=15,              # ADD: Limit tree depth
    min_samples_split=20,      # ADD: Require more samples to split
    min_samples_leaf=10,       # ADD: Require more samples per leaf
    max_features='sqrt',       # ADD: Use sqrt of features per split
    random_state=RANDOM_STATE,
    n_jobs=-1,
    class_weight="balanced_subsample"
)

# Also add cross-validation:
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy')
print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
```

---

### 3. **String Formatting Error in Report Generation**

**Location**: Cell 11 (Report Generation)

**Problem**:
```python
report.append(f"- File: `{pd}`\n")  # BUG: This prints the pandas module, not filename
```

**Why This Is a Problem**:
- `pd` is the pandas library import, not the filename
- The report will show something like `<module 'pandas' from ...>` instead of the actual CSV file name

**Solution**:
```python
# At the top of Cell 11, define the actual filename:
csv_filename = "California_Fire_Incidents.csv"

# Then use it in the report:
report.append(f"- File: `{csv_filename}`\n")
```

---

## Moderate Issues

### 4. **Missing Data Validation**

**Location**: Cell 1 (Data Loading)

**Problem**:
- No validation that the CSV file exists before attempting to read it
- No handling of potential file reading errors

**Solution**:
```python
import os

# Add validation before loading:
csv_path = "California_Fire_Incidents.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Data file not found: {csv_path}")

try:
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"Successfully loaded {len(df)} rows from {csv_path}")
except Exception as e:
    print(f"Error loading CSV: {e}")
    raise
```

---

### 5. **Inconsistent DataFrame Usage**

**Location**: Multiple cells (Cells 3, 4, 5)

**Problem**:
```python
# Cell 3 creates 'work' from df.copy()
work = df.copy()

# Cell 4 performs EDA on original 'df' instead of 'work'
# This creates inconsistency - features are added to df but model uses work
if "Counties" in df.columns:  # Should use work
    df["CountyPrimary"] = ...
```

**Why This Is a Problem**:
- EDA in Cell 4 modifies `df` by adding CountyPrimary, StartMonth, Season
- But the modeling pipeline uses `work` (later renamed to `fe`)
- Creates confusion about which DataFrame has which features

**Solution**:
```python
# Cell 4 - Change all 'df' references to 'work':
if "Counties" in work.columns:  # Changed from df
    work["CountyPrimary"] = work["Counties"].astype(str).str.split(",").str[0].str.strip()

if "Started" in work.columns:  # Changed from df
    work["Started"] = pd.to_datetime(work["Started"], errors="coerce")
    work["StartMonth"] = work["Started"].dt.month
    # ... etc

# Then perform EDA on work:
plt.figure(figsize=(10,6))
sns.countplot(data=work, x="CountyPrimary", ...)  # Changed from df
```

---

### 6. **No Missing Data Strategy Documentation**

**Location**: Cell 7 (Preprocessing)

**Problem**:
```python
# Median imputation for numeric features
("num", Pipeline(steps=[
    ("impute", SimpleImputer(strategy="median"))
]), candidate_numeric),
```

**Why This Is a Problem**:
- Some features have >90% missing values (as shown in Cell 2)
- Using median imputation on highly sparse features can introduce bias
- No threshold for dropping features with too much missing data

**Solution**:
```python
# After Cell 2, add missing data threshold:

MISSING_THRESHOLD = 0.7  # Drop features with >70% missing

# Filter features based on missingness:
missing_pct = fe[candidate_numeric].isna().mean()
valid_numeric_features = missing_pct[missing_pct < MISSING_THRESHOLD].index.tolist()

print(f"Dropped {len(candidate_numeric) - len(valid_numeric_features)} features due to >70% missing data")
print(f"Dropped features: {set(candidate_numeric) - set(valid_numeric_features)}")

# Use valid_numeric_features instead of candidate_numeric
```

---

### 7. **Hardcoded File Paths**

**Location**: Cells 1 and 9

**Problem**:
```python
df = pd.read_csv("California_Fire_Incidents.csv", low_memory=False)  # Relative path
fi_path = "rf_feature_importances.csv"  # Saves to current directory
REPORT_PATH = "rf_report.md"  # Saves to current directory
```

**Why This Is a Problem**:
- Notebook will fail if run from a different directory
- Output files scattered in potentially unknown locations
- Not portable or reproducible

**Solution**:
```python
# At the top of the notebook, define a project structure:
import os
from pathlib import Path

# Define paths relative to notebook location
NOTEBOOK_DIR = Path(__file__).parent if '__file__' in locals() else Path.cwd()
DATA_DIR = NOTEBOOK_DIR / "data"
OUTPUT_DIR = NOTEBOOK_DIR / "output"

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(exist_ok=True)

# Use in cells:
df = pd.read_csv(DATA_DIR / "California_Fire_Incidents.csv", low_memory=False)
fi_path = OUTPUT_DIR / "rf_feature_importances.csv"
REPORT_PATH = OUTPUT_DIR / "rf_report.md"
```

---

### 8. **No Model Persistence**

**Location**: Throughout the notebook

**Problem**:
- The trained model is not saved anywhere
- Cannot reuse the model without re-running entire notebook
- No versioning or tracking of model artifacts

**Solution**:
```python
# After Cell 7 (model training), add:
import joblib
from datetime import datetime

# Save the trained pipeline
model_filename = OUTPUT_DIR / f"rf_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
joblib.dump(pipe, model_filename)
print(f"Model saved to: {model_filename}")

# To load later:
# loaded_model = joblib.load(model_filename)
```

---

## Minor Issues

### 9. **Unclear Season Encoding**

**Location**: Cell 5 (Feature Engineering)

**Problem**:
```python
def month_to_season(m):
    if pd.isna(m): return np.nan
    m = int(m)
    return 0 if m in [12,1,2] else (1 if m in [3,4,5] else (2 if m in [6,7,8] else 3))
```

**Why This Is a Problem**:
- Nested ternary operators are hard to read
- No documentation of what 0, 1, 2, 3 represent
- Differs from the named season mapping in Cell 4

**Solution**:
```python
def month_to_season(m):
    """
    Convert month number to season code.
    0=Winter, 1=Spring, 2=Summer, 3=Fall
    """
    if pd.isna(m):
        return np.nan

    m = int(m)
    season_map = {
        12: 0, 1: 0, 2: 0,   # Winter
        3: 1, 4: 1, 5: 1,    # Spring
        6: 2, 7: 2, 8: 2,    # Summer
        9: 3, 10: 3, 11: 3   # Fall
    }
    return season_map.get(m, np.nan)
```

---

### 10. **Inconsistent Try-Except for sklearn Version Handling**

**Location**: Cells 7 and 10

**Problem**:
```python
# This pattern is repeated twice:
try:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
```

**Why This Is a Problem**:
- Code duplication violates DRY principle
- Should be a reusable function

**Solution**:
```python
# At the top after imports, add:
def create_onehot_encoder():
    """Create OneHotEncoder compatible with different sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # Fallback for older sklearn versions
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

# Then use it in cells:
ohe = create_onehot_encoder()
ohe2 = create_onehot_encoder()
```

---

### 11. **Magic Numbers Without Explanation**

**Location**: Multiple cells

**Problem**:
```python
work["LargeFire"] = (work["AcresBurned"] >= 1000).astype(int)  # Why 1000?
TOP_N = 25  # Why 25?
test_size=0.2  # Why 20%?
n_estimators=500  # Why 500?
```

**Solution**:
```python
# Define constants at the top with explanations:

# Target definition: Large fires are those burning >= 1000 acres
# This threshold is based on CAL FIRE's definition of "large incidents"
LARGE_FIRE_THRESHOLD_ACRES = 1000

# Model configuration
TEST_SIZE = 0.2  # 80/20 train-test split is standard
N_ESTIMATORS = 500  # Increased from sklearn default (100) for better performance
TOP_FEATURES_TO_SHOW = 25  # Display top 25 features in importance plot

# Then use them:
work["LargeFire"] = (work["AcresBurned"] >= LARGE_FIRE_THRESHOLD_ACRES).astype(int)
```

---

### 12. **Missing Exploratory Data Analysis for Target Variable**

**Location**: Cell 4 (EDA)

**Problem**:
- EDA analyzes features but never examines the distribution of AcresBurned in relation to the target threshold
- No analysis of class imbalance (only 18.68% positive class)

**Solution**:
```python
# Add after Cell 3:

# Analyze target distribution
print(f"Target Distribution:")
print(f"  Small fires (<1000 acres): {(work['LargeFire']==0).sum()} ({(work['LargeFire']==0).mean():.2%})")
print(f"  Large fires (>=1000 acres): {(work['LargeFire']==1).sum()} ({(work['LargeFire']==1).mean():.2%})")

# Visualize the threshold
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(work[work['AcresBurned'] < 10000]['AcresBurned'], bins=50, alpha=0.7)
plt.axvline(x=1000, color='red', linestyle='--', linewidth=2, label='Large Fire Threshold')
plt.xlabel('Acres Burned')
plt.ylabel('Frequency')
plt.title('Distribution of AcresBurned (filtered <10000 for visibility)')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(np.log10(work['AcresBurned'] + 1), bins=50, alpha=0.7)
plt.axvline(x=np.log10(1000), color='red', linestyle='--', linewidth=2, label='Large Fire Threshold')
plt.xlabel('Log10(Acres Burned + 1)')
plt.ylabel('Frequency')
plt.title('Log-scale Distribution')
plt.legend()
plt.tight_layout()
plt.show()
```

---

### 13. **No Threshold Tuning for Class Imbalance**

**Location**: Cell 8 (Evaluation)

**Problem**:
- With only 18.68% positive class, using default 0.5 threshold is suboptimal
- The model might benefit from adjusting the decision threshold
- No exploration of precision-recall tradeoffs

**Solution**:
```python
# After Cell 8, add threshold tuning:

from sklearn.metrics import precision_recall_curve

# Find optimal threshold based on F1 score
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)

optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

print(f"\nOptimal threshold: {optimal_threshold:.3f}")
print(f"At optimal threshold:")
print(f"  Precision: {precisions[optimal_idx]:.3f}")
print(f"  Recall: {recalls[optimal_idx]:.3f}")
print(f"  F1 Score: {f1_scores[optimal_idx]:.3f}")

# Apply optimal threshold
y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
print("\nClassification Report with Optimal Threshold:")
print(classification_report(y_test, y_pred_optimal, digits=3))
```

---

### 14. **Date Parsing Warning**

**Location**: Cell 3

**Problem**:
```python
# This shows minimum date as 1969-12-31 16:00:00+00:00
# This is likely Unix epoch (timestamp 0) indicating missing/invalid dates
Date coverage: 1969-12-31 16:00:00+00:00 → 2019-11-25 19:59:12+00:00
```

**Solution**:
```python
# After parsing dates in Cell 3:
for c in ["Started", "Updated"]:
    if c in work.columns:
        work[c] = pd.to_datetime(work[c], errors="coerce")

        # Filter out invalid dates (Unix epoch or before 1900)
        epoch_date = pd.Timestamp('1970-01-01', tz='UTC')
        invalid_dates = work[c] < pd.Timestamp('1900-01-01', tz='UTC')

        if invalid_dates.any():
            print(f"Warning: {invalid_dates.sum()} invalid dates found in {c}, setting to NaT")
            work.loc[invalid_dates, c] = pd.NaT

# Then show valid date range:
valid_dates = work["Started"].dropna()
if len(valid_dates) > 0:
    print(f"Valid date range: {valid_dates.min()} → {valid_dates.max()}")
```

---

### 15. **No Random State Verification**

**Location**: Cell 0

**Problem**:
```python
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
```

**Why This Is Incomplete**:
- Sets numpy random seed but doesn't set Python's random module
- sklearn has its own random state (handled via parameter)
- Not comprehensive for full reproducibility

**Solution**:
```python
import random

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

# Also set environment variable for hash randomization (optional but thorough)
import os
os.environ['PYTHONHASHSEED'] = str(RANDOM_STATE)

print(f"Random state set to {RANDOM_STATE} for reproducibility")
```

---

## Best Practice Improvements

### 16. **Add Logging Instead of Print Statements**

**Current**: Uses `print()` throughout
**Better**: Use logging module for better control

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('calfire_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Then use:
logger.info(f"Shape: {df.shape}")
logger.warning(f"Found {invalid_dates.sum()} invalid dates")
```

---

### 17. **Add Input Validation Function**

```python
def validate_dataframe(df, required_columns):
    """Validate that DataFrame has required columns and reasonable data."""
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if len(df) == 0:
        raise ValueError("DataFrame is empty")

    logger.info(f"Validation passed: {len(df)} rows, {len(df.columns)} columns")
    return True

# Use after loading data:
required_cols = ["AcresBurned", "Started", "Counties"]
validate_dataframe(df, required_cols)
```

---

### 18. **Add Model Performance Summary Function**

```python
def print_model_summary(y_true, y_pred, y_proba, model_name="Model"):
    """Print comprehensive model performance metrics."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, confusion_matrix
    )

    print(f"\n{'='*50}")
    print(f"{model_name} Performance Summary")
    print(f"{'='*50}")

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"Confusion Matrix:")
    print(f"  TN: {tn:4d}  FP: {fp:4d}")
    print(f"  FN: {fn:4d}  TP: {tp:4d}")
    print()

    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1 Score': f1_score(y_true, y_pred, zero_division=0),
        'ROC AUC': roc_auc_score(y_true, y_proba)
    }

    for metric, value in metrics.items():
        print(f"{metric:12s}: {value:.4f}")

    print(f"{'='*50}\n")

    return metrics
```

---

## Summary of All Issues

### Critical (Fix Immediately):
1. ✅ **Data Leakage**: Using post-hoc features (DurationDays, PersonnelInvolved, etc.)
2. ✅ **Overfitting**: 99.9% train vs 88.4% test accuracy gap
3. ✅ **String Formatting Bug**: `{pd}` instead of filename in report

### Moderate (Should Fix):
4. ✅ Missing file existence validation
5. ✅ Inconsistent DataFrame usage (df vs work)
6. ✅ No missing data threshold strategy
7. ✅ Hardcoded file paths
8. ✅ No model persistence

### Minor (Nice to Have):
9. ✅ Unclear season encoding with nested ternary
10. ✅ Duplicated OneHotEncoder try-except
11. ✅ Magic numbers without explanation
12. ✅ Missing target variable EDA
13. ✅ No threshold tuning for imbalanced classes
14. ✅ Invalid date handling (1969 dates)
15. ✅ Incomplete random state setting

### Best Practices:
16. ✅ Use logging instead of print
17. ✅ Add input validation
18. ✅ Add model performance summary function

---

## Priority Fix Order

1. **HIGHEST PRIORITY**: Remove data leakage features - this makes the model practically useless
2. **HIGH**: Add overfitting controls (max_depth, cross-validation)
3. **HIGH**: Fix string formatting bug in report
4. **MEDIUM**: Add file validation and proper path handling
5. **MEDIUM**: Fix DataFrame consistency issues
6. **LOW**: All other improvements

---

## Estimated Impact

| Issue | Current Impact | After Fix |
|-------|---------------|-----------|
| Data Leakage | Model unusable for prediction | Realistic 64% AUC performance |
| Overfitting | Poor generalization | More reliable predictions |
| Missing Validation | Crashes on bad input | Graceful error handling |
| Code Quality | Hard to maintain/extend | Professional, maintainable code |

---

## Conclusion

The code demonstrates good knowledge of machine learning workflows but has **critical flaws** that make the current model unsuitable for production use. The most serious issue is data leakage - the model achieves high accuracy by using information that would not be available at prediction time.

After implementing these fixes, you will have:
- A realistic model (lower accuracy but actually predictive)
- Production-ready code with proper validation
- Maintainable and reproducible pipeline
- Clear documentation of limitations

The good news is that all issues are fixable, and the overall structure of the notebook is sound.
