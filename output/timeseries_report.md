# CALFIRE Time Series Forecasting

Generated: 2026-07-22 21:15:29

Monthly series, 2013-01-01 to 2019-11-01 (83 months). Backtested on the last 12 months, then refit on all data to forecast 12 months forward.

## Monthly fire count

| Model | Backtest MAE | Backtest RMSE |
|---|---|---|
| AR **(best)** | 15.0 | 19.2 |
| Prophet | 18.5 | 20.1 |
| ARIMA | 16.5 | 26.5 |

Best backtest fit: **AR**

## Monthly acres burned

Heavy-tailed: a handful of mega-fires (e.g. Camp Fire, Mendocino Complex) dominate total acreage in the months they occur, so error here is naturally much larger than for fire *count*. Read this as "which model tracks the trend," not a precise acreage forecast.

| Model | Backtest MAE | Backtest RMSE |
|---|---|---|
| Prophet **(best)** | 235885.8 | 355583.5 |
| ARIMA | 451426.2 | 897253.7 |
| AR | 1317953.4 | 1984377.2 |

Best backtest fit: **Prophet**

