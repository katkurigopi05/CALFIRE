# CALFIRE Time Series Forecasting

Generated: 2026-07-22 22:56:02

Monthly series, 2013-01-01 to 2019-11-01 (83 months). Backtested on the last 12 months, then refit on all data to forecast 12 months forward. Naive (repeat last month) and Seasonal Naive (repeat same month last year) are included as baselines — a model only earns its complexity if it beats them. Model selection below is by RMSE; MAPE is reported for reference but blows up in low-fire-activity months (small denominator), so it's not the deciding metric.

## Monthly fire count

| Model | Backtest MAE | Backtest RMSE | Backtest MAPE |
|---|---|---|---|
| AR **(best)** | 15.0 | 19.2 | 123.7% |
| Prophet | 18.5 | 20.1 | 300.9% |
| SeasonalNaive | 16.4 | 22.4 | 95.2% |
| Naive | 20.6 | 22.5 | 279.4% |
| ARIMA | 16.5 | 26.5 | 81.0% |

Best backtest fit: **AR** — beats the best naive baseline (RMSE 22.4).

## Monthly acres burned

Heavy-tailed: a handful of mega-fires (e.g. Camp Fire, Mendocino Complex) dominate total acreage in the months they occur, so error here is naturally much larger than for fire *count*. Read this as "which model tracks the trend," not a precise acreage forecast.

| Model | Backtest MAE | Backtest RMSE | Backtest MAPE |
|---|---|---|---|
| Naive **(best)** | 330205.2 | 332587.3 | 145785.1% |
| Prophet | 235885.8 | 355583.5 | 69924.4% |
| SeasonalNaive | 323649.2 | 728899.0 | 2205.5% |
| ARIMA | 451426.2 | 897253.7 | 10546.0% |
| AR | 1317953.4 | 1984377.2 | 36864.0% |

Best backtest fit: **Naive** — none of AR/ARIMA/Prophet beat this naive baseline here, so the honest takeaway is "last month"/"same month last year" is as good a forecast as anything fancier.

