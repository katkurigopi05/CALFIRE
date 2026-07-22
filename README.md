# CALFIRE
California wildfire classification prediction using ML

Predicts whether a California fire will become "large", using only
information available at ignition time — and how large, since "large" is
tested at several acreage cutoffs rather than a single fixed line. Also
includes time-series forecasting of fire activity itself (monthly count and
acres burned).

## Contents

- `California_Fire_Incidents.csv` — CAL FIRE historical incident data (2013-2019)
- `CALFIRE(RF) RN7945.ipynb` — original notebook
- `CALFIRE_improved.py` — first-pass leak-free baseline (see `CODE_REVIEW.md`)
- `scripts/model_common.py` — shared feature/data-loading helpers (geo cleaning, county history rate, geo clustering, the model bake-off)
- `scripts/train_model.py` — flagship single-threshold model (≥1,000 ac): cyclical date encoding, dry-season flag, leak-free county fire-history rate, geo clustering, optional weather features, RandomForest/HistGradientBoosting/LogisticRegression bake-off. Run: `python scripts/train_model.py`
- `scripts/threshold_analysis.py` — runs the same pipeline at 10/50/100/500/1,000 acres to see which "large fire" definitions the data actually has a learnable pattern for (see `output/threshold_analysis_report.md`). Powers `predict.py`'s full risk profile and the dashboard's threshold chart.
- `scripts/timeseries_analysis.py` — aggregates incidents into monthly series (count, acres burned) and forecasts with AR, ARIMA (SARIMAX), and Prophet, picking whichever backtests best (see `output/timeseries_report.md`). Feeds the dashboard's forecast panel.
- `scripts/fetch_weather.py` — pulls historical daily weather (temp/precip/wind/ET0/humidity) per incident from the free Open-Meteo Archive API. **Must be run somewhere with open internet access** — writes `weather_data.csv`, which `train_model.py` and `threshold_analysis.py` pick up automatically if present.
- `scripts/predict.py` — CLI risk tool: given a county + date (and optionally weather or `--fetch-weather`), prints large-fire probability at every threshold from `threshold_analysis.py`.
- `scripts/build_dashboard_data.py` — builds `dashboard/index.html`, a self-contained offline dashboard (trends, seasonality, county impact, geographic scatter, threshold comparison, forecasts) from the CSV plus whichever of the above have been run.

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
pip install pandas numpy scikit-learn statsmodels prophet joblib requests

python scripts/train_model.py            # flagship >=1,000 ac model
python scripts/threshold_analysis.py     # multi-threshold pattern analysis + risk-profile models
python scripts/timeseries_analysis.py    # AR / ARIMA / Prophet forecasts
python scripts/build_dashboard_data.py   # regenerates dashboard/index.html from all of the above

python scripts/predict.py --county Riverside --date 2026-08-15
```

## What the multi-threshold analysis found

Re-running the same model at 10/50/100/500/1,000 acres shows the "large fire"
line matters: at 10 acres, 98% of recorded incidents already qualify (CAL
FIRE's public dataset mostly logs already-notable fires), so there's no real
minority class to learn from. Signal strengthens steadily from there — test
ROC-AUC goes from ~0.62 at 10 acres up to ~0.72 at 1,000 acres. See
`output/threshold_analysis_report.md` for the full breakdown.
