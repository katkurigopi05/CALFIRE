# CALFIRE Multi-Threshold Pattern Analysis

Generated: 2026-07-22 22:56:00

Same leak-free feature pipeline and model bake-off, run separately at each acreage cutoff, to see where the data actually carries a learnable signal for "this fire will get big."

| Threshold (ac) | % large | Best model | CV ROC-AUC | Test ROC-AUC | Test F1 |
|---|---|---|---|---|---|
| 10 | 98.3% (1439/1464) | LogisticRegression | 0.477 | 0.622 | 0.792 |
| 50 | 67.5% (988/1464) | RandomForest | 0.645 | 0.582 | 0.711 |
| 100 | 50.1% (734/1464) | RandomForest | 0.626 | 0.650 | 0.592 |
| 500 | 23.6% (346/1464) | RandomForest | 0.657 | 0.692 | 0.446 |
| 1,000 | 18.6% (272/1464) | RandomForest | 0.641 | 0.724 | 0.434 |

## Detail per threshold

### >= 10 acres
Model bake-off (CV ROC-AUC):
- LogisticRegression: 0.4773 ± 0.1463
- RandomForest: 0.3898 ± 0.1445
- HistGradientBoosting: 0.3765 ± 0.1515

Holdout metrics (LogisticRegression):
- Accuracy: 0.6587
- Precision: 0.9896
- Recall: 0.6597
- F1 Score: 0.7917
- ROC AUC: 0.6222

### >= 50 acres
Model bake-off (CV ROC-AUC):
- RandomForest: 0.6450 ± 0.0577
- LogisticRegression: 0.6323 ± 0.0753
- HistGradientBoosting: 0.6129 ± 0.0512

Holdout metrics (RandomForest):
- Accuracy: 0.6143
- Precision: 0.7202
- Recall: 0.7020
- F1 Score: 0.7110
- ROC AUC: 0.5818

### >= 100 acres
Model bake-off (CV ROC-AUC):
- RandomForest: 0.6262 ± 0.0489
- LogisticRegression: 0.6208 ± 0.0470
- HistGradientBoosting: 0.6137 ± 0.0496

Holdout metrics (RandomForest):
- Accuracy: 0.6007
- Precision: 0.6071
- Recall: 0.5782
- F1 Score: 0.5923
- ROC AUC: 0.6497

### >= 500 acres
Model bake-off (CV ROC-AUC):
- RandomForest: 0.6572 ± 0.0702
- HistGradientBoosting: 0.6314 ± 0.0964
- LogisticRegression: 0.6129 ± 0.0604

Holdout metrics (RandomForest):
- Accuracy: 0.7031
- Precision: 0.3977
- Recall: 0.5072
- F1 Score: 0.4459
- ROC AUC: 0.6924

### >= 1,000 acres
Model bake-off (CV ROC-AUC):
- RandomForest: 0.6406 ± 0.1117
- LogisticRegression: 0.6258 ± 0.1189
- HistGradientBoosting: 0.5894 ± 0.1205

Holdout metrics (RandomForest):
- Accuracy: 0.7509
- Precision: 0.3733
- Recall: 0.5185
- F1 Score: 0.4341
- ROC AUC: 0.7244
