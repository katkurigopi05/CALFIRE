"""
CALFIRE 100+ year historical trend analysis (1878-2025).

Uses the CAL FIRE FRAP historical fire perimeter dataset (California_
Historic_Fire_Perimeters.csv — see README for how to get it) to look at
long-run annual trends in fire count and acres burned. Unlike the point
incident dataset (2013-2019, has lat/lon), this is polygon perimeter data
with no location field, so it feeds trend/forecast analysis only, not the
spatial risk model.

Per CAL FIRE's own data-quality note, pre-1950 records are much sparser
(lost records, smaller fires never captured, looser criteria) — modeling
uses the 1950-2025 window; the full 1878-2025 series is charted for context
but not backtested.

Model bake-off (annual granularity, so no seasonal component): Naive, Linear
Trend, Quadratic Trend, ETS (Holt linear trend), and ARIMA — same
accuracy-comparison discipline as scripts/timeseries_analysis.py (a model
must beat Naive to be worth trusting).

Usage: python scripts/historical_trends.py
"""
import json
import logging
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error

from model_common import load_historic_perimeters

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")
logger = logging.getLogger("CALFIRE-historical")

MODEL_START_YEAR = 1950  # CAL FIRE's own cutoff for more-reliable collection
BACKTEST_YEARS = 10
FORECAST_YEARS = 10

ROOT = Path(__file__).parent.parent
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
REPORT_PATH = OUTPUT_DIR / "historical_trends_report.md"
DASHBOARD_JSON_PATH = ROOT / "dashboard" / "historical.json"


def build_annual_series(df):
    annual = df.groupby("Year").agg(count=("Acres", "count"), acres=("Acres", "sum"))
    full_index = pd.RangeIndex(annual.index.min(), annual.index.max() + 1)
    return annual.reindex(full_index, fill_value=0)


def fit_naive(train, steps):
    return np.full(steps, train.iloc[-1])


def fit_trend(train, steps, degree):
    x = np.arange(len(train))
    X = np.vander(x, degree + 1)
    model = sm.OLS(train.values, X).fit()
    future_x = np.arange(len(train), len(train) + steps)
    future_X = np.vander(future_x, degree + 1)
    return model.predict(future_X)


def fit_ets(train, steps):
    model = ExponentialSmoothing(train.values, trend="add", damped_trend=True).fit()
    return model.forecast(steps)


def fit_arima(train, steps):
    model = SARIMAX(train.values, order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    fc = model.get_forecast(steps=steps)
    return fc.predicted_mean, fc.conf_int()


def safe_mape(actual, pred):
    actual, pred = np.asarray(actual), np.asarray(pred)
    mask = actual != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100)


def backtest(series, label):
    train, test = series.iloc[:-BACKTEST_YEARS], series.iloc[-BACKTEST_YEARS:]
    preds = {
        "Naive": fit_naive(train, BACKTEST_YEARS),
        "LinearTrend": fit_trend(train, BACKTEST_YEARS, 1),
        "QuadraticTrend": fit_trend(train, BACKTEST_YEARS, 2),
        "ETS": fit_ets(train, BACKTEST_YEARS),
        "ARIMA": np.asarray(fit_arima(train, BACKTEST_YEARS)[0]),
    }
    scores = {}
    for name, pred in preds.items():
        pred = np.clip(pred, 0, None)
        mae = mean_absolute_error(test.values, pred)
        rmse = np.sqrt(mean_squared_error(test.values, pred))
        mape = safe_mape(test.values, pred)
        scores[name] = {"mae": mae, "rmse": rmse, "mape": mape}
        logger.info("[%s] %-15s backtest MAE=%.2f RMSE=%.2f MAPE=%.1f%%", label, name, mae, rmse, mape)
    return scores


def forecast_forward(series, label):
    naive = np.clip(fit_naive(series, FORECAST_YEARS), 0, None)
    linear = np.clip(fit_trend(series, FORECAST_YEARS, 1), 0, None)
    quad = np.clip(fit_trend(series, FORECAST_YEARS, 2), 0, None)
    ets = np.clip(fit_ets(series, FORECAST_YEARS), 0, None)
    arima_mean, arima_ci = fit_arima(series, FORECAST_YEARS)
    arima_mean = np.clip(arima_mean, 0, None)

    future_years = list(range(series.index[-1] + 1, series.index[-1] + 1 + FORECAST_YEARS))
    return {
        "history": {"years": series.index.tolist(), "values": series.values.tolist()},
        "forecast": {
            "years": future_years,
            "Naive": naive.tolist(), "LinearTrend": linear.tolist(),
            "QuadraticTrend": quad.tolist(), "ETS": ets.tolist(),
            "ARIMA": arima_mean.tolist(),
            "ARIMA_lower": np.clip(arima_ci[:, 0], 0, None).tolist(),
            "ARIMA_upper": arima_ci[:, 1].tolist(),
        },
    }


def main():
    df = load_historic_perimeters()
    logger.info("Loaded %d fires, %d-%d", len(df), df["Year"].min(), df["Year"].max())

    full_annual = build_annual_series(df)
    model_annual = full_annual[full_annual.index >= MODEL_START_YEAR]
    logger.info("Full history: %d years | Modeling window (%d+): %d years",
                len(full_annual), MODEL_START_YEAR, len(model_annual))

    report = [
        "# CALFIRE 100+ Year Historical Trend Analysis\n\n",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n",
        f"Source: CAL FIRE FRAP historical fire perimeter dataset, "
        f"{full_annual.index.min()}-{full_annual.index.max()} "
        f"({len(df):,} recorded fires). Pre-{MODEL_START_YEAR} records are "
        f"sparser (lost records, looser collection criteria per CAL FIRE's "
        f"own metadata), so modeling/backtesting uses {MODEL_START_YEAR}-"
        f"{full_annual.index.max()} only; the full history is charted for "
        f"context but not fit.\n\n",
        "## Decade summary (full history)\n\n",
        "| Decade | Fires | Total acres |\n|---|---|---|\n",
    ]
    decade = df.groupby((df["Year"] // 10) * 10).agg(count=("Acres", "count"), acres=("Acres", "sum"))
    for dec, row in decade.iterrows():
        report.append(f"| {dec}s | {int(row['count']):,} | {row['acres']:,.0f} |\n")

    report.append("\n## Top causes (full history)\n\n| Cause | Fires |\n|---|---|\n")
    for cause, count in df["CauseName"].value_counts().head(10).items():
        report.append(f"| {cause} | {count:,} |\n")

    forecast_payload = {}
    for col, label in [("count", "Annual fire count"), ("acres", "Annual acres burned")]:
        series = model_annual[col].astype(float)
        report.append(f"\n## {label} ({MODEL_START_YEAR}-{full_annual.index.max()})\n\n")
        scores = backtest(series, label)
        best_model = min(scores, key=lambda k: scores[k]["rmse"])
        report.append("| Model | Backtest MAE | Backtest RMSE | Backtest MAPE |\n|---|---|---|---|\n")
        for name, s in sorted(scores.items(), key=lambda kv: kv[1]["rmse"]):
            marker = " **(best)**" if name == best_model else ""
            mape_str = f"{s['mape']:.1f}%" if not np.isnan(s["mape"]) else "n/a"
            report.append(f"| {name}{marker} | {s['mae']:.1f} | {s['rmse']:.1f} | {mape_str} |\n")
        ranked = sorted(scores.items(), key=lambda kv: kv[1]["rmse"])
        runner_up_name, runner_up = ranked[1]
        margin = (runner_up["rmse"] - scores[best_model]["rmse"]) / scores[best_model]["rmse"]

        report.append(f"\nBest backtest fit: **{best_model}**")
        if best_model == "Naive":
            report.append(" — none of the trend/ETS/ARIMA models beat flat repetition of the last value here.\n\n")
        elif margin < 0.05:
            report.append(
                f" — only {margin*100:.1f}% better RMSE than {runner_up_name}, and MAPE disagrees on the "
                f"ranking ({best_model} {scores[best_model]['mape']:.1f}% vs {runner_up_name} "
                f"{runner_up['mape']:.1f}%). Treat this as a toss-up, not a clear win.\n\n"
            )
        else:
            report.append(f" — beats Naive (RMSE {scores['Naive']['rmse']:.1f}).\n\n")

        fc = forecast_forward(series, label)
        fc["best_model"] = best_model
        forecast_payload[col] = fc
        forecast_payload[col]["full_history"] = {
            "years": full_annual.index.tolist(),
            "values": full_annual[col].astype(float).values.tolist(),
        }

    REPORT_PATH.write_text("".join(report))
    logger.info("Report saved -> %s", REPORT_PATH)

    forecast_payload["decade_summary"] = [
        {"decade": f"{int(dec)}s", "count": int(row["count"]), "acres": float(row["acres"])}
        for dec, row in decade.iterrows()
    ]
    forecast_payload["cause_breakdown"] = [
        {"cause": "Unknown" if cause == "Unknown/Unidentified" else cause, "count": int(count)}
        for cause, count in df["CauseName"].value_counts().head(10).items()
    ]
    forecast_payload["model_start_year"] = MODEL_START_YEAR

    DASHBOARD_JSON_PATH.parent.mkdir(exist_ok=True)
    DASHBOARD_JSON_PATH.write_text(json.dumps(forecast_payload, default=float))
    logger.info("Dashboard data saved -> %s", DASHBOARD_JSON_PATH)


if __name__ == "__main__":
    main()
