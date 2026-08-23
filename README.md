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
- `CALFIRE_Administrative_Units.csv` — the 27 CAL FIRE administrative units (name, code, Northern/Southern region), from the same Hub's "CAL FIRE Administrative Units" dataset. Only covers CAL FIRE's own jurisdiction — ~46% of fires in the historical dataset are on federal land (USFS/NPS/BLM) with different unit codes this doesn't have, so those are labeled "Federal/Other Agency" rather than guessed at.
- `CALFIRE(RF) RN7945.ipynb` — original notebook
- `CALFIRE_improved.py` — first-pass leak-free baseline (see `CODE_REVIEW.md`)
- `scripts/model_common.py` — shared feature/data-loading helpers (geo cleaning, county history rate, geo clustering, the model bake-off)
- `scripts/train_model.py` — flagship single-threshold model (≥1,000 ac): cyclical date encoding, dry-season flag, leak-free county fire-history rate, geo clustering, optional weather features, RandomForest/HistGradientBoosting/LogisticRegression bake-off. Run: `python scripts/train_model.py`
- `scripts/threshold_analysis.py` — runs the same pipeline at 10/50/100/500/1,000 acres to see which "large fire" definitions the data actually has a learnable pattern for (see `output/threshold_analysis_report.md`). Powers `predict.py`'s full risk profile and the dashboard's threshold chart.
- `scripts/timeseries_analysis.py` — aggregates incidents into monthly series (count, acres burned) and forecasts with Naive, Seasonal Naive, a trend+season regression (tslm-style, per the companion Time-Series repo's methodology), AR, ARIMA (SARIMAX), and Prophet, picking whichever backtests best (see `output/timeseries_report.md`). Feeds the dashboard's 12-month forecast panel. TrendSeason currently wins both series — it beats AR on fire count and turns the acres-burned series from a "toss-up vs Naive" into a real, beaten-baseline forecast.
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
pip install -r requirements.txt

python scripts/run_pipeline.py           # runs everything below, in order, stops on first failure
python scripts/predict.py --county Riverside --date 2026-08-15
```

`run_pipeline.py` is just the five steps below run in sequence, for
convenience — run them individually instead if you only need one:

```bash
python scripts/train_model.py            # flagship >=1,000 ac model
python scripts/threshold_analysis.py     # multi-threshold pattern analysis + risk-profile models
python scripts/timeseries_analysis.py    # AR / ARIMA / Prophet forecasts (2013-2019, monthly)
python scripts/historical_trends.py      # 1878-2025 annual trend analysis + forecast
python scripts/build_dashboard_data.py   # regenerates dashboard/index.html from all of the above
```

## CI

`.github/workflows/pipeline.yml` runs the same five steps as
`run_pipeline.py` (every script except `fetch_weather.py`, which needs live
internet access) on every push and PR to `main`, as separate steps rather
than one call to `run_pipeline.py` so a failure points at the exact script
that broke instead of just "the pipeline." A change that breaks any script
gets caught automatically instead of only being noticed the next time
someone runs it by hand.

## What the multi-threshold analysis found

Re-running the same model at 10/50/100/500/1,000 acres shows the "large fire"
line matters: at 10 acres, 98% of recorded incidents already qualify (CAL
FIRE's public dataset mostly logs already-notable fires), so there's no real
minority class to learn from. Signal strengthens steadily from there — test
ROC-AUC goes from ~0.62 at 10 acres up to ~0.72 at 1,000 acres — filling in
the gap with 75/150/200/300/750-acre cutoffs shows this is a steady climb
across the whole 50-1,000 range, not one sharp inflection point. See
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

Breaking causes down by decade shows a real shift: **Power Line-caused
fires went from essentially 0 before 1950 to 161 in the 2010s**, and Vehicle-
and Equipment-caused fires show a similar rise — consistent with aging grid
infrastructure and development pushing further into wildland, not just more
fires overall (Lightning, the natural-cause baseline, stays comparatively
flat).

`CALFIRE_Administrative_Units.csv` lets us label the historical dataset's
`Unit ID` field with real unit names and Northern/Southern region — but only
for the 46% of fires on CAL FIRE-managed land; the rest (federal land) are
explicitly labeled "Federal/Other Agency" rather than mis-mapped. Within
CAL FIRE's own jurisdiction, Southern units have burned somewhat more total
acreage than Northern (9.3M vs 8.0M acres) across the full 148 years.

## Where and when fires peak, and destructiveness beyond acreage

The dashboard's month × county heatmap (from the 2013-2019 point dataset)
shows most counties peak in the Jun-Aug dry season, but a few — Ventura,
Napa, Sonoma — peak in Oct/Nov instead, consistent with wind-driven fall
fire season (Santa Ana/Diablo winds) rather than pure dryness.

Acreage alone doesn't capture destructiveness: the dashboard also ranks
fires by structures destroyed per 1,000 acres (small-footprint fires that
hit dense areas hard) and by fatalities. Note the structures/fatalities
fields are only populated for ~11%/~1% of fires respectively — treat these
as "worst known cases," not a complete accounting.

## Running it

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
python CALFIRE_improved.py            # the cleaned-up pipeline
jupyter notebook "CALFIRE(RF) RN7945.ipynb"   # the exploratory version
```

The dashboard is static — open `dashboard/index.html` directly in a browser.
It reads the committed `dashboard/*.json` files (`data`, `forecast`,
`historical`, `threshold_summary`), so it works offline with no server.

`.github/workflows/pipeline.yml` runs the pipeline on push, which is also the
quickest reference for the exact Python version and install steps used.

See `CODE_REVIEW.md` and `QUICK_FIXES.md` for known rough edges.

## Libraries & Methods

Scanned every `.py`/`.ipynb` file (11 files, 2,484 lines).

**Modeling** — `sklearn.ensemble.RandomForestClassifier` is the core
classifier, wrapped in a `Pipeline` with `ColumnTransformer`,
`SimpleImputer`, `OneHotEncoder` and `OrdinalEncoder`; evaluated with
`cross_val_score` under `StratifiedKFold`, scored with `mean_absolute_error`
and `mean_squared_error`.

**Forecasting** — three different time-series models are actually compared,
not just one: `statsmodels.tsa.statespace.SARIMAX`, `statsmodels.tsa.holtwinters
.ExponentialSmoothing`, `statsmodels.tsa.ar_model.AutoReg`, and `prophet.Prophet`.

**Data** — pandas, numpy, `joblib` for model persistence.

**Visualization** — matplotlib, seaborn.

**Own modules** — `model_common` (`clean_lat_lon`, `load_historic_perimeters`,
`add_calfire_unit_labels`, `load_dates_and_acres`) and a `fetch_weather` module
that pulls live weather via `requests`.
