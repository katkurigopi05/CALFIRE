# CALFIRE
California wildfire classification prediction using ML

Predicts whether a California fire will become a "large" incident
(≥1,000 acres burned), using only information available at ignition time.

## Contents

- `California_Fire_Incidents.csv` — CAL FIRE historical incident data (2013-2019)
- `CALFIRE(RF) RN7945.ipynb` — original notebook
- `CALFIRE_improved.py` — first-pass leak-free baseline (see `CODE_REVIEW.md`)
- `scripts/model_common.py` — shared feature/cleaning helpers (geo cleaning, county history rate, geo clustering)
- `scripts/train_model.py` — v2 model: cyclical date encoding, dry-season flag, leak-free county fire-history rate, geo clustering, optional weather features, and a RandomForest/HistGradientBoosting/LogisticRegression bake-off. Run: `python scripts/train_model.py`
- `scripts/fetch_weather.py` — pulls historical daily weather (temp/precip/wind/ET0/humidity) per incident from the free Open-Meteo Archive API. **Must be run somewhere with open internet access** — writes `weather_data.csv`, which `train_model.py` picks up automatically if present.
- `scripts/predict.py` — CLI risk tool: given a county + date (and optionally weather or `--fetch-weather`), scores large-fire probability with the trained model.
- `scripts/build_dashboard_data.py` — builds `dashboard/index.html`, a self-contained offline dashboard (trends, seasonality, county impact, geographic scatter) from the CSV.

## Getting real weather into the model

This dataset only has date + location for each fire — no weather. To add real
weather features (which meaningfully improves accuracy — see
`output/model_v2_report.md`):

```bash
pip install pandas requests
python scripts/fetch_weather.py   # run somewhere with internet access
python scripts/train_model.py     # picks up weather_data.csv automatically
```

## Quick start

```bash
pip install pandas numpy scikit-learn joblib requests
python scripts/train_model.py
python scripts/predict.py --county Riverside --date 2026-08-15
python scripts/build_dashboard_data.py   # regenerates dashboard/index.html
```
