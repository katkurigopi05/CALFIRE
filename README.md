# CALFIRE
California wildfire classification prediction using ML

Predicts whether a California fire will become "large", using only
information available at ignition time — and how large, since "large" is
tested at several acreage cutoffs rather than a single fixed line. Also
includes time-series forecasting of fire activity itself (monthly count and
acres burned).

## Contents

- `California_Fire_Incidents.csv` — CAL FIRE historical incident data (2013-2019), with lat/lon per fire
- `California_Historic_Fire_Perimeters.csv` — CAL FIRE FRAP historical fire perimeter dataset, 1878-2025 (23k+ fires). No lat/lon (it's polygon perimeters, not point incidents), but reliable Year/Alarm Date/Acres/Cause going back over a century. Downloaded from CNRA's ArcGIS Hub (`gis.data.cnra.ca.gov`) — "California Historical Fire Perimeters" → "California Fire Perimeters (all)" → Download → CSV.
- `CALFIRE(RF) RN7945.ipynb` — original notebook
- `CALFIRE_improved.py` — first-pass leak-free baseline (see `CODE_REVIEW.md`)
- `scripts/model_common.py` — shared feature/data-loading helpers (geo cleaning, county history rate, geo clustering, the model bake-off)
- `scripts/train_model.py` — flagship single-threshold model (≥1,000 ac): cyclical date encoding, dry-season flag, leak-free county fire-history rate, geo clustering, optional weather features, RandomForest/HistGradientBoosting/LogisticRegression bake-off. Run: `python scripts/train_model.py`
- `scripts/threshold_analysis.py` — runs the same pipeline at 10/50/100/500/1,000 acres to see which "large fire" definitions the data actually has a learnable pattern for (see `output/threshold_analysis_report.md`). Powers `predict.py`'s full risk profile and the dashboard's threshold chart.
- `scripts/timeseries_analysis.py` — aggregates incidents into monthly series (count, acres burned) and forecasts with Naive, Seasonal Naive, AR, ARIMA (SARIMAX), and Prophet, picking whichever backtests best (see `output/timeseries_report.md`). Feeds the dashboard's 12-month forecast panel.
- `scripts/historical_trends.py` — long-run annual trend analysis on `California_Historic_Fire_Perimeters.csv` (1878-2025). Backtests Naive, Linear/Quadratic Trend regression, ETS, and ARIMA on the 1950+ window (CAL FIRE's own cutoff for more-reliable collection), forecasts 10 years forward, and reports decade/cause breakdowns (see `output/historical_trends_report.md`). Feeds the dashboard's "148 Years of California Wildfire History" panel.
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
python scripts/timeseries_analysis.py    # AR / ARIMA / Prophet forecasts (2013-2019, monthly)
python scripts/historical_trends.py      # 1878-2025 annual trend analysis + forecast
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

## What 148 years of data shows

Extending to the 1878-2025 FRAP perimeter dataset: annual fire count and
acres burned both show a real long-run upward trend (worst decade for acres
burned is the 2020s, more than 100x the 1900s), though pre-1950 records are
known to be incomplete per CAL FIRE's own metadata, so `historical_trends.py`
only backtests/forecasts on 1950-2025. Even there, results are mixed — the
count series clearly beats a naive "repeat last year" baseline (Quadratic
Trend, RMSE 126 vs Naive's 156), but for acres burned the "win" is only ~1%
better than Naive and the metrics disagree on the ranking, so treat that one
as a toss-up rather than a real forecast. See
`output/historical_trends_report.md` for the full breakdown, including
decade and fire-cause tables.
