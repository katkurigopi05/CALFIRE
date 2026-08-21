"""
CALFIRE time-series forecasting: AR, ARIMA (SARIMAX), Prophet, and a
trend+season regression.

Aggregates incidents into monthly series (fire count, total acres burned) and
fits several forecasting models to each, so we can read off seasonality (CA's
annual fire season) and compare which model actually predicts held-out months
best before trusting any forward forecast.

Naive and Seasonal Naive are included as baselines that the fancier models
must actually beat — a model that loses to "repeat last month" or "repeat
the same month last year" isn't earning its complexity. TrendSeason (linear
trend + seasonal month dummies, i.e. tslm(~ trend + season)) is included as
a classical benchmark alongside them. Methodology follows the
accuracy-comparison structure used in a companion time-series project:
github.com/katkurigopi05/Time-Series.

Usage: python scripts/timeseries_analysis.py
"""
import json
import logging
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import statsmodels.api as sm

from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

from model_common import load_dates_and_acres

warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")
logger = logging.getLogger("CALFIRE-timeseries")

BACKTEST_MONTHS = 12
FORECAST_MONTHS = 12
SEASONAL_PERIOD = 12

ROOT = Path(__file__).parent.parent
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
REPORT_PATH = OUTPUT_DIR / "timeseries_report.md"
FORECAST_JSON_PATH = ROOT / "dashboard" / "forecast.json"


def build_monthly_series():
    df = load_dates_and_acres()
    monthly = df.set_index("Started").resample("MS").agg(
        count=("AcresBurned", "count"), acres=("AcresBurned", "sum"),
    )
    return monthly


def fit_naive(train, steps):
    return np.full(steps, train.iloc[-1])


def fit_seasonal_naive(train, steps, period=SEASONAL_PERIOD):
    last_cycle = train.iloc[-period:].values
    return np.array([last_cycle[i % period] for i in range(steps)])


def _trend_season_design(dates, trend_values):
    """tslm(~ trend + season)-style design matrix: a linear trend plus 11
    month dummies (12th absorbed into the intercept)."""
    months = pd.Categorical(dates.month, categories=range(1, 13))
    X = pd.get_dummies(months, prefix="m", drop_first=True).astype(float)
    X.insert(0, "trend", trend_values)
    return sm.add_constant(X, has_constant="add")


def fit_trend_season(train, steps):
    """Regression with linear trend + seasonal month dummies, matching
    tslm(train ~ trend + season) from the reference time-series methodology
    (github.com/katkurigopi05/Time-Series)."""
    X_train = _trend_season_design(train.index, np.arange(len(train)))
    model = sm.OLS(train.values, X_train.values).fit()

    future_dates = pd.date_range(train.index[-1] + pd.offsets.MonthBegin(1), periods=steps, freq="MS")
    X_future = _trend_season_design(future_dates, np.arange(len(train), len(train) + steps))
    X_future = X_future[X_train.columns]
    return model.predict(X_future.values)


def fit_ar(train, steps):
    model = AutoReg(train, lags=SEASONAL_PERIOD).fit()
    return model.predict(start=len(train), end=len(train) + steps - 1)


def fit_sarimax(train, steps):
    model = SARIMAX(
        train, order=(1, 1, 1), seasonal_order=(1, 0, 1, SEASONAL_PERIOD),
        enforce_stationarity=False, enforce_invertibility=False,
    ).fit(disp=False)
    fc = model.get_forecast(steps=steps)
    return fc.predicted_mean, fc.conf_int()


def fit_prophet(train_series, steps):
    from prophet import Prophet
    dfp = pd.DataFrame({"ds": train_series.index.tz_localize(None), "y": train_series.values})
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
                interval_width=0.8)
    m.fit(dfp)
    future = m.make_future_dataframe(periods=steps, freq="MS")
    fc = m.predict(future)
    return fc.tail(steps)


def safe_mape(actual, pred):
    """Mean absolute percentage error, computed only over months with a
    nonzero actual (MAPE is undefined at zero and fire count/acres both hit
    zero some months)."""
    actual, pred = np.asarray(actual), np.asarray(pred)
    mask = actual != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100)


def backtest_series(series, name):
    train, test = series.iloc[:-BACKTEST_MONTHS], series.iloc[-BACKTEST_MONTHS:]
    results = {
        "Naive": fit_naive(train, BACKTEST_MONTHS),
        "SeasonalNaive": fit_seasonal_naive(train, BACKTEST_MONTHS),
        "TrendSeason": np.asarray(fit_trend_season(train, BACKTEST_MONTHS)),
        "AR": np.asarray(fit_ar(train, BACKTEST_MONTHS)),
        "ARIMA": np.asarray(fit_sarimax(train, BACKTEST_MONTHS)[0]),
        "Prophet": fit_prophet(train, BACKTEST_MONTHS)["yhat"].values,
    }

    scores = {}
    for model_name, pred in results.items():
        pred = np.clip(pred, 0, None)
        mae = mean_absolute_error(test.values, pred)
        rmse = np.sqrt(mean_squared_error(test.values, pred))
        mape = safe_mape(test.values, pred)
        scores[model_name] = {"mae": mae, "rmse": rmse, "mape": mape}
        logger.info("[%s] %-13s backtest MAE=%.2f RMSE=%.2f MAPE=%.1f%%", name, model_name, mae, rmse, mape)

    return scores


def forecast_series(series, name):
    """Refit each model (including the naive baselines) on the FULL series
    and forecast forward, so whichever model wins the backtest can be looked
    up by name for the dashboard."""
    ar_pred = np.clip(fit_ar(series, FORECAST_MONTHS), 0, None)
    sarimax_pred, sarimax_ci = fit_sarimax(series, FORECAST_MONTHS)
    sarimax_pred = np.clip(sarimax_pred, 0, None)
    prophet_fc = fit_prophet(series, FORECAST_MONTHS)
    naive_pred = np.clip(fit_naive(series, FORECAST_MONTHS), 0, None)
    seasonal_naive_pred = np.clip(fit_seasonal_naive(series, FORECAST_MONTHS), 0, None)
    trend_season_pred = np.clip(fit_trend_season(series, FORECAST_MONTHS), 0, None)

    future_dates = pd.date_range(series.index[-1] + pd.offsets.MonthBegin(1), periods=FORECAST_MONTHS, freq="MS")

    return {
        "history": {"dates": [d.strftime("%Y-%m-%d") for d in series.index], "values": series.values.tolist()},
        "forecast": {
            "dates": [d.strftime("%Y-%m-%d") for d in future_dates],
            "Naive": naive_pred.tolist(),
            "SeasonalNaive": seasonal_naive_pred.tolist(),
            "TrendSeason": trend_season_pred.tolist(),
            "AR": ar_pred.tolist(),
            "ARIMA": sarimax_pred.tolist(),
            "ARIMA_lower": np.clip(sarimax_ci.iloc[:, 0].values, 0, None).tolist(),
            "ARIMA_upper": sarimax_ci.iloc[:, 1].values.tolist(),
            "Prophet": np.clip(prophet_fc["yhat"].values, 0, None).tolist(),
            "Prophet_lower": np.clip(prophet_fc["yhat_lower"].values, 0, None).tolist(),
            "Prophet_upper": prophet_fc["yhat_upper"].values.tolist(),
        },
    }


def main():
    monthly = build_monthly_series()
    logger.info("Monthly series: %d months (%s to %s)", len(monthly), monthly.index.min().date(), monthly.index.max().date())

    report = [
        "# CALFIRE Time Series Forecasting\n\n",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n",
        f"Monthly series, {monthly.index.min().date()} to {monthly.index.max().date()} "
        f"({len(monthly)} months). Backtested on the last {BACKTEST_MONTHS} months, "
        f"then refit on all data to forecast {FORECAST_MONTHS} months forward. "
        f"Naive (repeat last month) and Seasonal Naive (repeat same month last "
        f"year) are included as baselines — a model only earns its complexity "
        f"if it beats them. Model selection below is by RMSE; MAPE is reported "
        f"for reference but blows up in low-fire-activity months (small "
        f"denominator), so it's not the deciding metric.\n\n",
    ]

    forecast_payload = {}
    for col, label in [("count", "Monthly fire count"), ("acres", "Monthly acres burned")]:
        series = monthly[col].astype(float)
        report.append(f"## {label}\n\n")
        if col == "acres":
            report.append(
                "Heavy-tailed: a handful of mega-fires (e.g. Camp Fire, Mendocino "
                "Complex) dominate total acreage in the months they occur, so "
                "error here is naturally much larger than for fire *count*. "
                "Read this as \"which model tracks the trend,\" not a precise "
                "acreage forecast.\n\n"
            )
        scores = backtest_series(series, label)
        report.append("| Model | Backtest MAE | Backtest RMSE | Backtest MAPE |\n|---|---|---|---|\n")
        best_model = min(scores, key=lambda k: scores[k]["rmse"])
        for model_name, s in sorted(scores.items(), key=lambda kv: kv[1]["rmse"]):
            marker = " **(best)**" if model_name == best_model else ""
            mape_str = f"{s['mape']:.1f}%" if not np.isnan(s["mape"]) else "n/a"
            report.append(f"| {model_name}{marker} | {s['mae']:.1f} | {s['rmse']:.1f} | {mape_str} |\n")
        report.append(f"\nBest backtest fit: **{best_model}**")
        if best_model in ("Naive", "SeasonalNaive"):
            report.append(
                " — none of AR/ARIMA/Prophet beat this naive baseline here, "
                "so the honest takeaway is \"last month\"/\"same month last "
                "year\" is as good a forecast as anything fancier.\n\n"
            )
        else:
            naive_rmse = min(scores["Naive"]["rmse"], scores["SeasonalNaive"]["rmse"])
            report.append(f" — beats the best naive baseline (RMSE {naive_rmse:.1f}).\n\n")

        forecast_payload[col] = forecast_series(series, label)
        forecast_payload[col]["best_model"] = best_model
        forecast_payload[col]["backtest_scores"] = scores

    REPORT_PATH.write_text("".join(report))
    logger.info("Report saved -> %s", REPORT_PATH)

    FORECAST_JSON_PATH.parent.mkdir(exist_ok=True)
    FORECAST_JSON_PATH.write_text(json.dumps(forecast_payload, default=float))
    logger.info("Forecast data saved -> %s", FORECAST_JSON_PATH)


if __name__ == "__main__":
    main()
