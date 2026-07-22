# CALFIRE Model v2 — Results

Generated: 2026-07-22 21:08:27

Weather features: no (weather_data.csv not found)

## Model bake-off (5-fold CV ROC-AUC)
- RandomForest: 0.6406 ± 0.1117
- LogisticRegression: 0.6258 ± 0.1189
- HistGradientBoosting: 0.5894 ± 0.1205

**Selected: RandomForest**

## Holdout test performance
- Accuracy: 0.7509
- Precision: 0.3733
- Recall: 0.5185
- F1 Score: 0.4341
- ROC AUC: 0.7244

- Optimal classification threshold: 0.527

## Features
- Numeric: ['StartYear', 'DOY_sin', 'DOY_cos', 'StartWeekday', 'DrySeason', 'CountyHistoricalRate']
- Categorical: ['Status']
- Geo cluster: ['Latitude', 'Longitude']
