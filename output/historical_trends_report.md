# CALFIRE 100+ Year Historical Trend Analysis

Generated: 2026-07-22 23:26:01

Source: CAL FIRE FRAP historical fire perimeter dataset, 1878-2025 (23,257 recorded fires). Pre-1950 records are sparser (lost records, looser collection criteria per CAL FIRE's own metadata), so modeling/backtesting uses 1950-2025 only; the full history is charted for context but not fit.

## Decade summary (full history)

| Decade | Fires | Total acres |
|---|---|---|
| 1870s | 1 | 59,469 |
| 1890s | 7 | 36,404 |
| 1900s | 70 | 89,124 |
| 1910s | 1,324 | 1,671,693 |
| 1920s | 1,479 | 2,925,772 |
| 1930s | 1,157 | 1,605,232 |
| 1940s | 1,155 | 1,959,130 |
| 1950s | 1,819 | 2,844,981 |
| 1960s | 1,276 | 2,075,063 |
| 1970s | 1,879 | 2,606,294 |
| 1980s | 2,206 | 3,078,897 |
| 1990s | 1,992 | 3,348,844 |
| 2000s | 2,905 | 6,528,778 |
| 2010s | 3,429 | 6,907,328 |
| 2020s | 2,558 | 8,880,678 |

## Top causes (full history)

| Cause | Fires |
|---|---|
| Unknown/Unidentified | 10,437 |
| Lightning | 3,648 |
| Miscellaneous | 3,533 |
| Equipment Use | 1,455 |
| Arson | 1,018 |
| Debris | 806 |
| Vehicle | 639 |
| Power Line | 506 |
| Campfire | 411 |
| Smoking | 341 |

## How causes have shifted, decade by decade

| Cause | 1900s | 1910s | 1920s | 1930s | 1940s | 1950s | 1960s | 1970s | 1980s | 1990s | 2000s | 2010s | 2020s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Unknown/Unidentified | 70 | 916 | 777 | 500 | 606 | 1272 | 848 | 1166 | 951 | 672 | 489 | 1079 | 1083 |
| Lightning | 0 | 107 | 167 | 146 | 141 | 181 | 136 | 203 | 363 | 372 | 842 | 676 | 314 |
| Miscellaneous | 0 | 240 | 440 | 351 | 335 | 256 | 196 | 218 | 328 | 230 | 317 | 437 | 185 |
| Equipment Use | 0 | 1 | 3 | 6 | 9 | 31 | 24 | 47 | 151 | 192 | 334 | 386 | 271 |
| Arson | 0 | 7 | 6 | 23 | 1 | 6 | 9 | 71 | 186 | 183 | 238 | 147 | 141 |
| Debris | 0 | 43 | 67 | 71 | 23 | 25 | 24 | 48 | 67 | 68 | 139 | 124 | 107 |
| Vehicle | 0 | 0 | 0 | 1 | 0 | 0 | 4 | 0 | 6 | 40 | 158 | 211 | 219 |
| Power Line | 0 | 0 | 0 | 0 | 0 | 2 | 2 | 2 | 8 | 73 | 128 | 161 | 130 |

## CAL FIRE unit / region breakdown

Only covers CAL FIRE's own jurisdiction (46% of fires) — the rest are on federal land (USFS/NPS/BLM), grouped as "Federal/Other Agency" since this lookup has no unit codes for those.

| Region | Fires | Total acres |
|---|---|---|
| Federal/Other Agency | 12,648 | 27,324,277 |
| Southern | 7,445 | 9,257,753 |
| Northern | 3,164 | 8,035,657 |

### Top CAL FIRE units by acres burned

| Unit | Fires | Total acres |
|---|---|---|
| Sonoma-Lake Napa Unit | 635 | 2,325,932 |
| Los Angeles Unit | 2,034 | 2,177,460 |
| Butte Unit | 239 | 1,873,478 |
| Riverside Unit | 892 | 1,452,406 |
| Ventura County | 441 | 1,234,453 |
| San Benito-Monterey Unit | 399 | 869,935 |
| Tehama-Glenn Unit | 299 | 834,357 |
| Santa Clara Unit | 310 | 813,339 |
| Shasta-Trinity Unit | 284 | 670,099 |
| Madera-Mariposa-Merced Unit | 343 | 507,210 |
| Kern County | 814 | 496,762 |
| Mendocino Unit | 362 | 490,699 |
| San Luis Obispo Unit | 515 | 447,672 |
| San Bernardino Unit | 255 | 403,655 |
| Fresno-Kings Unit | 367 | 393,753 |

## Annual fire count (1950-2025)

| Model | Backtest MAE | Backtest RMSE | Backtest MAPE |
|---|---|---|---|
| QuadraticTrend **(best)** | 102.2 | 125.6 | 22.3% |
| Naive | 119.8 | 156.0 | 24.0% |
| LinearTrend | 130.6 | 166.2 | 26.4% |
| ARIMA | 133.7 | 170.5 | 27.0% |
| ETS | 134.1 | 170.7 | 27.1% |

Best backtest fit: **QuadraticTrend** — beats Naive (RMSE 156.0).


## Annual acres burned (1950-2025)

| Model | Backtest MAE | Backtest RMSE | Backtest MAPE |
|---|---|---|---|
| QuadraticTrend **(best)** | 883386.6 | 1269755.2 | 90.8% |
| Naive | 870779.7 | 1279783.9 | 82.1% |
| ETS | 873934.1 | 1325925.4 | 71.6% |
| ARIMA | 871646.5 | 1361318.4 | 62.8% |
| LinearTrend | 873951.4 | 1366270.0 | 62.7% |

Best backtest fit: **QuadraticTrend** — only 0.8% better RMSE than Naive, and MAPE disagrees on the ranking (QuadraticTrend 90.8% vs Naive 82.1%). Treat this as a toss-up, not a clear win.

