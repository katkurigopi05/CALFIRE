# Quick Fix Guide - Top 5 Critical Issues

## 🚨 Issue #1: DATA LEAKAGE (CRITICAL)

**Problem**: Model uses features only available AFTER the fire (99.9% train accuracy is a red flag)

**Location**: Cell 5 and Cell 7

**Quick Fix**:
```python
# REMOVE these features from your model - they cause data leakage:
leakage_features = [
    "DurationDays",        # Only known after fire ends
    "PercentContained",    # Only known during firefighting
    "PersonnelInvolved",   # Only known during response
    "Engines",             # Deployed during fire
    "Dozers",              # Deployed during fire
    "AirTankers",          # Deployed during fire
    "Helicopters",         # Deployed during fire
    "CrewsInvolved",       # Deployed during fire
    "WaterTenders"         # Deployed during fire
]

# ONLY use these features (available at fire start):
prediction_features = [
    "ArchiveYear", "StartYear", "StartMonth",
    "StartDOY", "StartWeekday", "StartSeason",
    "CountyPrimary", "Status"
]
```

---

## 🚨 Issue #2: OVERFITTING (CRITICAL)

**Problem**: 99.9% train accuracy vs 88.4% test accuracy = severe overfitting

**Location**: Cell 7

**Quick Fix**:
```python
rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=15,              # ADD THIS
    min_samples_split=20,      # ADD THIS
    min_samples_leaf=10,       # ADD THIS
    max_features='sqrt',       # ADD THIS
    random_state=RANDOM_STATE,
    n_jobs=-1,
    class_weight="balanced_subsample"
)
```

---

## 🚨 Issue #3: BUG IN REPORT (HIGH)

**Problem**: Report prints pandas module instead of filename

**Location**: Cell 11, line with `{pd}`

**Quick Fix**:
```python
# WRONG:
report.append(f"- File: `{pd}`\n")

# CORRECT:
csv_filename = "California_Fire_Incidents.csv"
report.append(f"- File: `{csv_filename}`\n")
```

---

## 🚨 Issue #4: INCONSISTENT DATAFRAMES (MEDIUM)

**Problem**: Cell 4 modifies `df` but model uses `work`

**Location**: Cell 4

**Quick Fix**:
```python
# Change ALL occurrences in Cell 4 from:
if "Counties" in df.columns:
    df["CountyPrimary"] = ...

# TO:
if "Counties" in work.columns:
    work["CountyPrimary"] = ...

# And change plotting from:
sns.countplot(data=df, ...)

# TO:
sns.countplot(data=work, ...)
```

---

## 🚨 Issue #5: NO FILE VALIDATION (MEDIUM)

**Problem**: Code crashes if CSV doesn't exist

**Location**: Cell 1

**Quick Fix**:
```python
import os

csv_path = "California_Fire_Incidents.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Data file not found: {csv_path}")

df = pd.read_csv(csv_path, low_memory=False)
```

---

## Testing Your Fixes

After making these changes, you should see:

**Before Fixes**:
```
Train accuracy: 0.9992  (99.9% - TOO HIGH, overfitting)
Test  accuracy: 0.8838  (88.4%)
AUC ROC: 0.98+ (unrealistic due to leakage)
```

**After Fixes**:
```
Train accuracy: ~0.85   (More reasonable)
Test  accuracy: ~0.82   (Closer to train - better generalization)
AUC ROC: ~0.64          (Realistic for this problem)
```

**Note**: Lower accuracy after fixes is GOOD - it means you're solving the real problem, not cheating with data leakage!

---

## Implementation Order

1. **First**: Fix data leakage (Issue #1) - most critical
2. **Second**: Fix overfitting (Issue #2) - prevents memorization
3. **Third**: Fix report bug (Issue #3) - simple fix
4. **Fourth**: Fix DataFrame consistency (Issue #4)
5. **Fifth**: Add file validation (Issue #5)

For complete details on all 18 issues found, see `CODE_REVIEW.md`.
