# Retail demand forecast

A 1-day-ahead forecast of total units sold per day for a UK online gift shop, built to help plan stock and staff, with honest baselines and a small Streamlit demo.

## Results at a glance

- On the last 3 months (78 open days), gradient boosting on log units gets **MAE 7,113** units/day against **7,339** for "same day last week", about **3% better**. A 7-day mean baseline gets 7,835.
- The gain is small because a few very large orders cannot be predicted from past daily totals, and forecasting 3 or 7 days ahead does not beat the weekly baseline.
- Found and fixed a cleaning bug: cancelled orders were counted as demand. Matching cancellations to their original orders removed 405,553 units of demand that never happened.

![Test period: actual vs predicted](images/actual_vs_predicted.png)

Stack: pandas, scikit-learn (`HistGradientBoostingRegressor`, `TimeSeriesSplit`), matplotlib, Streamlit, pytest.

## Problem and data

The shop needs a rough idea of how many units it will ship tomorrow. Data: the [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) dataset, real transactions from a UK online gift retailer from 1 Dec 2009 to 9 Dec 2011. It has 1,067,371 rows (one per invoice line), and 995,377 remain after cleaning. That gives 738 calendar days, of which 504 open days are used for training and 78 for testing (2011-09-09 to 2011-12-08).

![Daily units](images/daily_units.png)

## How it works

1. **Clean:** drop exact duplicate rows, remove each cancellation together with the order it cancelled, drop the other cancellations, rows with quantity or price of 0 or less, and non-product stock codes (postage, fees, manual entries, gift vouchers).
2. **Cut the partial last day:** the file stops at 12:50 on 9 Dec 2011, while other recent weekdays run to about 18:00, so the data ends on 8 Dec.
3. **Daily series:** total units per day, with every calendar day filled in and 0 on closed days (Saturdays, Christmas), so "7 days ago" is always the same weekday.
4. **Features from the past only:** day of week, month, units 1, 7 and 14 days ago, and the mean of the previous 7 and 28 days.
5. **Split by time:** the last 91 calendar days are the test set. Closed days are left out of training and scoring, because predicting 0 on a Saturday is trivial.
6. **Model:** log or raw target is chosen by 5-fold time-series cross-validation on the training period only. Two quantile models add a 10 to 90% range.

## Results

| Method | MAE | RMSE | MAPE |
|---|---|---|---|
| Same day last week | 7,339 | 9,540 | 32.1% |
| Mean of last 7 days | 7,835 | 10,030 | 35.3% |
| Gradient boosting, raw units | 7,446 | 9,583 | 32.0% |
| Gradient boosting, log units (saved) | 7,113 | 9,318 | 29.7% |

**Log vs raw target.** In cross-validation on the training period, log units won on average (MAE 5,347 against 6,197) and in each of the 5 folds. But the margin per fold ranged from 40 to 1,976 units, and the fold standard deviation (about 1,340) is larger than the average gap (850), so it is a modest win, not a decisive one.

**Forecast range.** If the 10 to 90% range were well calibrated, about 80% of test days would fall inside it. Only 60% do (21 above, 10 below), with an average width of 12,401 units. It is a rough guide, not a reliable interval.

**Forecast horizon.** One model per horizon, a forecast made at the end of day t for day t+h, scored on the same 78 test days. Baselines use only what is known on day t.

| Days ahead | Model MAE | Same weekday last week | 7-day mean at origin | Gain vs best baseline |
|---|---|---|---|---|
| 1 | 7,113 | 7,339 | 7,835 | 3.1% |
| 3 | 7,459 | 7,339 | 8,218 | -1.6% |
| 7 | 7,279 | 7,339 | 8,358 | 0.8% |

The gain does not grow with the horizon. At 3 days the model is slightly worse than "same weekday last week" and at 7 days it is about level, so the weekly pattern carries most of what past totals can tell us.

![Horizons](images/horizons.png)

## What I found

- **Cancellation bug.** The first version dropped cancellation rows but kept the orders they cancelled. One example: an order of 80,995 units of "PAPER CRAFT , LITTLE BIRDIE" on 9 Dec 2011, cancelled 12 minutes later by invoice C581484. In the first version it was the worst forecast error. The fix removes, for each cancellation, the most recent earlier order line with the same customer, stock code and quantity. It matched 6,402 of 18,390 cancellations that have a Customer ID and removed 405,553 units. A test checks that both rows go. Results are not compared with the version before the fix, because the fix changed the test data itself.
- **Bank holidays** were tried as a feature, gave mixed cross-validation results, and were left out.

**Worst days are big underpredictions.** The 5 largest test errors are all days the model predicted far too low:

| Date | Actual | Predicted | Biggest single invoice |
|---|---|---|---|
| 2011-10-05 | 46,146 | 18,040 | 7,020 |
| 2011-09-20 | 42,401 | 18,493 | 11,816 |
| 2011-10-20 | 40,769 | 17,708 | 13,392 |
| 2011-09-13 | 36,492 | 14,362 | 7,824 |
| 2011-11-14 | 47,043 | 25,594 | 13,956 |

A single invoice of 7,020 to 13,956 units on these days is not visible in past daily totals, so the model cannot see it coming.

## Limitations

- The test set is one autumn and Christmas season, 78 days, so the numbers are noisy.
- The gain over the simple weekly baseline is about 3% at 1 day and disappears at 3 and 7 days.
- Only exact cancellations are matched. Partial cancellations and ones without a Customer ID are dropped without removing their original order, so some fake demand remains.
- It forecasts the whole shop's total, not per product, and has no price, promotion or customer features.

## Demo

![Streamlit app](images/app.png)

`streamlit run app.py` replays the 1-day-ahead forecast on a test day you pick: the forecast made from earlier days, what actually sold, the 10 to 90% range and the MAE against the baseline. It is not a live forecast of tomorrow.

## How to run

Needs Python 3.12+. Reading the Excel file takes about 30 seconds. `train.py` saves the model, metrics, test predictions and charts, and the app needs them, so run it first.

Linux or macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir -p data
curl -L -o data/retail.zip https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip
unzip data/retail.zip -d data
python train.py
pytest
streamlit run app.py
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1   # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt
New-Item -ItemType Directory -Force data
Invoke-WebRequest -Uri "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip" -OutFile data\retail.zip
Expand-Archive data\retail.zip -DestinationPath data
python train.py
pytest
streamlit run app.py
```

`pytest` runs 8 tests: cleaning, closed-day filling, no use of future values (including for the 3 and 7 day horizons), and checks on the saved model.

## Project structure

```text
train.py              cleaning, daily series, features, CV, models, baselines, horizons, charts
app.py                Streamlit demo that replays test days
tests/test_train.py   tests for cleaning, features, leakage and the saved model
pytest.ini            lets pytest import train.py from the repo root
requirements.txt      pinned Python packages
images/               charts and the app screenshot used in this README
```
