# CALFIRE Multi-Threshold Pattern Analysis

Generated: 2026-07-23 00:48:20

Same leak-free feature pipeline and model bake-off, run separately at each acreage cutoff, to see where the data actually carries a learnable signal for "this fire will get big."

| Threshold (ac) | % large | Best model | CV ROC-AUC | Test ROC-AUC | Test F1 |
|---|---|---|---|---|---|
| 10 | 98.3% (1439/1464) | LogisticRegression | 0.477 | 0.622 | 0.792 |
| 50 | 67.5% (988/1464) | RandomForest | 0.645 | 0.582 | 0.711 |
| 75 | 57.2% (838/1464) | RandomForest | 0.642 | 0.622 | 0.663 |
| 100 | 50.1% (734/1464) | RandomForest | 0.626 | 0.650 | 0.592 |
| 150 | 41.3% (605/1464) | RandomForest | 0.619 | 0.629 | 0.535 |
| 200 | 35.5% (520/1464) | RandomForest | 0.631 | 0.654 | 0.559 |
| 300 | 29.1% (426/1464) | RandomForest | 0.650 | 0.668 | 0.466 |
| 500 | 23.6% (346/1464) | RandomForest | 0.657 | 0.692 | 0.446 |
| 750 | 20.3% (297/1464) | RandomForest | 0.646 | 0.712 | 0.424 |
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

### >= 75 acres
Model bake-off (CV ROC-AUC):
- RandomForest: 0.6415 ± 0.0610
- LogisticRegression: 0.6309 ± 0.0542
- HistGradientBoosting: 0.6071 ± 0.0529

Holdout metrics (RandomForest):
- Accuracy: 0.6109
- Precision: 0.6588
- Recall: 0.6667
- F1 Score: 0.6627
- ROC AUC: 0.6225

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

### >= 150 acres
Model bake-off (CV ROC-AUC):
- RandomForest: 0.6195 ± 0.0992
- LogisticRegression: 0.6156 ± 0.0884
- HistGradientBoosting: 0.5955 ± 0.0811

Holdout metrics (RandomForest):
- Accuracy: 0.5973
- Precision: 0.5113
- Recall: 0.5620
- F1 Score: 0.5354
- ROC AUC: 0.6290

### >= 200 acres
Model bake-off (CV ROC-AUC):
- RandomForest: 0.6309 ± 0.0494
- HistGradientBoosting: 0.6156 ± 0.0863
- LogisticRegression: 0.6102 ± 0.0860

Holdout metrics (RandomForest):
- Accuracy: 0.6451
- Precision: 0.5000
- Recall: 0.6346
- F1 Score: 0.5593
- ROC AUC: 0.6540

### >= 300 acres
Model bake-off (CV ROC-AUC):
- RandomForest: 0.6500 ± 0.1044
- HistGradientBoosting: 0.6226 ± 0.0608
- LogisticRegression: 0.6080 ± 0.0717

Holdout metrics (RandomForest):
- Accuracy: 0.6792
- Precision: 0.4505
- Recall: 0.4824
- F1 Score: 0.4659
- ROC AUC: 0.6679

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

### >= 750 acres
Model bake-off (CV ROC-AUC):
- RandomForest: 0.6455 ± 0.0885
- HistGradientBoosting: 0.6268 ± 0.0590
- LogisticRegression: 0.6091 ± 0.0477

Holdout metrics (RandomForest):
- Accuracy: 0.7406
- Precision: 0.3836
- Recall: 0.4746
- F1 Score: 0.4242
- ROC AUC: 0.7116

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
