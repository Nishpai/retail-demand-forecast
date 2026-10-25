# Train a daily demand forecaster on UCI Online Retail II.
# Target: total units sold by the whole shop on a day.

from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_FILE = Path("data/online_retail_II.xlsx")
MODEL_FILE = Path("models/forecaster.joblib")
TEST_DAYS = 91  # roughly the last 3 months
# The file ends at 12:50 on 2011-12-09, while other weekdays run to about 18:00.
# That half day would look like a crash in demand, so data from that date on is cut.
LAST_FULL_DAY = pd.Timestamp("2011-12-08")
FEATURES = ["day_of_week", "month", "last", "lag_7", "lag_14", "mean_7", "mean_28"]
HORIZONS = [1, 3, 7]


def keep_rows(df, keep, reason):
    print(f"  {reason}: removed {(~keep).sum()}")
    return df[keep]


def remove_cancelled_originals(df):
    # A cancellation ("C" invoice) cancels an earlier order line. Remove the most
    # recent earlier line with the same customer, product and quantity.
    # Partial cancellations and cancellations without a Customer ID are not
    # matched; those cancellation rows are simply dropped later.
    is_cancel = df["Invoice"].astype(str).str.startswith("C")
    cancels = df[is_cancel & df["Customer ID"].notna()].sort_values("InvoiceDate")
    orders = df[~is_cancel & (df["Quantity"] > 0)].sort_values("InvoiceDate")
    # (customer, product, quantity) -> row labels of matching orders, oldest first
    groups = orders.groupby(["Customer ID", "StockCode", "Quantity"]).groups

    used = set()
    units = 0
    for _, c in cancels.iterrows():
        candidates = groups.get((c["Customer ID"], c["StockCode"], -c["Quantity"]), [])
        earlier = []
        for i in candidates:
            if i not in used and df.at[i, "InvoiceDate"] <= c["InvoiceDate"]:
                earlier.append(i)
        if earlier:
            used.add(earlier[-1])  # the most recent one
            units += -c["Quantity"]
    print(f"  cancellations matched to an original order: {len(used)} of {len(cancels)}, "
          f"removed {units} units")
    return df.drop(index=list(used))


def clean(df):
    df = keep_rows(df, ~df.duplicated(), "exact duplicate rows")
    df = remove_cancelled_originals(df)
    df = keep_rows(df, ~df["Invoice"].astype(str).str.startswith("C"), "cancelled invoices")
    df = keep_rows(df, (df["Quantity"] > 0) & (df["Price"] > 0), "quantity or price <= 0")
    # Real products have codes starting with 5 digits (e.g. 85123A).
    # The rest are postage, manual entries, fees, gift vouchers and so on.
    return keep_rows(df, df["StockCode"].astype(str).str.match(r"^\d{5}"), "non-product stock codes")


def make_daily(df):
    # The shop is closed on Saturdays and over Christmas, so those days have
    # no rows. Fill every calendar day with 0 so lag_7 means "same weekday last week".
    daily = df.groupby(df["InvoiceDate"].dt.normalize())["Quantity"].sum()
    all_days = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    return daily.reindex(all_days, fill_value=0).rename("units").to_frame()


def add_features(daily, h=1):
    # Each row is the day we want to forecast. The forecast is made h days earlier,
    # so "last" and the means can only use days up to h days before.
    # lag_7 and lag_14 are fine as long as h is 7 or less.
    out = daily[["units"]].copy()
    out["day_of_week"] = out.index.day_of_week
    out["month"] = out.index.month
    out["last"] = out["units"].shift(h)
    out["lag_7"] = out["units"].shift(7)
    out["lag_14"] = out["units"].shift(14)
    out["mean_7"] = out["units"].shift(h).rolling(7).mean()
    out["mean_28"] = out["units"].shift(h).rolling(28).mean()
    return out


def split(daily, cutoff):
    # Closed days are easy zeros, so leave them out of training and scoring.
    open_days = daily[daily["units"] > 0].dropna()
    return open_days[open_days.index <= cutoff], open_days[open_days.index > cutoff]


def fit_predict(train, rows, log_target, quantile=None):
    # quantile=None is the normal model; 0.1 or 0.9 gives one edge of the range.
    if quantile:
        model = HistGradientBoostingRegressor(random_state=42, loss="quantile", quantile=quantile)
    else:
        model = HistGradientBoostingRegressor(random_state=42)
    y = train["units"]
    if log_target:
        y = np.log1p(y)
    model.fit(train[FEATURES], y)
    pred = model.predict(rows[FEATURES])
    if log_target:
        pred = np.expm1(pred)
    return model, pred


def cv_mae(train, log_target):
    # Each fold trains on the past and checks on the next block of days.
    maes = []
    for fit_idx, val_idx in TimeSeriesSplit(n_splits=5).split(train):
        val = train.iloc[val_idx]
        _, pred = fit_predict(train.iloc[fit_idx], val, log_target)
        maes.append(mean_absolute_error(val["units"], pred))
    return np.array(maes)


def report(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs(y_true - y_pred) / y_true) * 100
    print(f"{name:<22} MAE {mae:8.0f}   RMSE {rmse:8.0f}   MAPE {mape:5.1f}%")
    return {"method": name, "MAE": round(mae), "RMSE": round(rmse), "MAPE %": round(mape, 1)}


def load_clean_data():
    if not DATA_FILE.exists():
        raise SystemExit("Data not found. See README for the download command.")
    # The file has two sheets, one per year. Stack them.
    sheets = pd.read_excel(DATA_FILE, sheet_name=None)
    df = pd.concat(sheets.values(), ignore_index=True)
    print(f"Raw rows: {len(df)}\nCleaning:")
    df = clean(df)
    df = keep_rows(df, df["InvoiceDate"] < LAST_FULL_DAY + pd.Timedelta(days=1),
                   "rows on the partial last day (2011-12-09)")
    print(f"Clean rows: {len(df)}")
    return df


def main():
    df = load_clean_data()
    daily = add_features(make_daily(df))
    print(f"Dates: {daily.index[0].date()} to {daily.index[-1].date()} ({len(daily)} days)")
    cutoff = daily.index.max() - pd.Timedelta(days=TEST_DAYS)
    train, test = split(daily, cutoff)
    print(f"Train: {len(train)} open days, test: {len(test)} open days "
          f"({test.index.min().date()} to {test.index.max().date()})")

    # Raw units vs log1p(units). Pick by CV on train only, never by test.
    raw_cv = cv_mae(train, log_target=False)
    log_cv = cv_mae(train, log_target=True)
    for name, maes in [("raw", raw_cv), ("log", log_cv)]:
        print(f"5-fold time CV MAE, {name} target: {maes.mean():.0f} +/- {maes.std():.0f}, "
              f"folds {maes.round().astype(int).tolist()}")
    use_log = log_cv.mean() < raw_cv.mean()
    print(f"Chosen target: {'log1p(units)' if use_log else 'raw units'}")

    y = test["units"]
    print("\nTest results (closed days excluded, so MAPE is safe):")
    rows = [report("Same day last week", y, test["lag_7"]),
            report("Mean of last 7 days", y, test["mean_7"])]
    raw_model, raw_pred = fit_predict(train, test, log_target=False)
    rows.append(report("GB, raw target", y, raw_pred))
    log_model, log_pred = fit_predict(train, test, log_target=True)
    rows.append(report("GB, log target", y, log_pred))
    if use_log:
        model, pred = log_model, log_pred
    else:
        model, pred = raw_model, raw_pred

    # 10-90% range. The quantile models use the same target as the saved model.
    # On the log target this is still fine: expm1 keeps the order of values, so
    # the 10th percentile of log units maps back to the 10th percentile of units.
    lower_model, lower = fit_predict(train, test, use_log, quantile=0.1)
    upper_model, upper = fit_predict(train, test, use_log, quantile=0.9)
    inside = (y >= lower) & (y <= upper)
    print(f"\n10-90% range: {inside.mean():.0%} of test days inside (target 80%), "
          f"average width {(upper - lower).mean():.0f} units")

    out = pd.DataFrame({"actual": y, "predicted": pred.round(), "baseline": test["lag_7"],
                        "lower": lower.round(), "upper": upper.round()})
    out["error"] = out["actual"] - out["predicted"]
    worst_days(df, out)

    MODEL_FILE.parent.mkdir(exist_ok=True)
    joblib.dump({"model": model, "log_target": use_log,
                 "lower": lower_model, "upper": upper_model}, MODEL_FILE)
    daily[["units"]].to_csv("data/daily.csv")
    out.to_csv("data/test_predictions.csv")
    pd.DataFrame(rows).to_csv("models/metrics.csv", index=False)
    save_charts(daily, out)

    horizons = horizon_table(daily, cutoff)
    print("\nForecast horizon (MAE on the same test days):")
    print(horizons.to_string(index=False))
    horizons.to_csv("models/horizons.csv", index=False)
    save_horizon_chart(horizons)
    print(f"\nSaved {MODEL_FILE}, data/daily.csv, data/test_predictions.csv, models/metrics.csv, "
          "models/horizons.csv")


def horizon_table(daily, cutoff):
    # One model per horizon, scored on the same open test days.
    results = []
    for h in HORIZONS:
        train, test = split(add_features(daily, h), cutoff)
        use_log = cv_mae(train, True).mean() < cv_mae(train, False).mean()
        _, pred = fit_predict(train, test, use_log)
        y = test["units"]
        model_mae = mean_absolute_error(y, pred)
        week_mae = mean_absolute_error(y, test["lag_7"])
        mean_mae = mean_absolute_error(y, test["mean_7"])
        best = min(week_mae, mean_mae)
        results.append({
            "horizon": h,
            "test days": len(test),
            "target": "log" if use_log else "raw",
            "model": round(model_mae),
            "same weekday last week": round(week_mae),
            "7-day mean at origin": round(mean_mae),
            "gain vs best baseline %": round(100 * (best - model_mae) / best, 1),
        })
    return pd.DataFrame(results)


def worst_days(df, out):
    # Error analysis: the 5 test days with the largest absolute error.
    worst = out.loc[out["error"].abs().sort_values(ascending=False).index[:5]]
    print("\nWorst 5 test days:")
    print(worst.to_string())
    invoice_units = df.groupby([df["InvoiceDate"].dt.normalize(), "Invoice"])["Quantity"].sum()
    for day in worst.index:
        print(f"  {day.date()}: biggest single invoice {invoice_units.loc[day].max()} units")


def save_charts(daily, out):
    Path("images").mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 4))
    daily["units"].rolling(7).mean().plot(ax=ax, label="7-day average")
    ax.set(title="Daily units sold, Dec 2009 to Dec 2011", ylabel="units")
    ax.legend()
    fig.savefig("images/daily_units.png", dpi=100, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.fill_between(out.index, out["lower"], out["upper"], color="tab:orange", alpha=0.2,
                    label="10-90% range")
    ax.plot(out.index, out["actual"], ".-", label="actual")
    ax.plot(out.index, out["predicted"], ".-", label="gradient boosting")
    ax.plot(out.index, out["baseline"], ".-", label="same day last week")
    ax.set(title="Test period: actual vs predicted", ylabel="units")
    ax.legend()
    fig.savefig("images/actual_vs_predicted.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def save_horizon_chart(horizons):
    fig, ax = plt.subplots(figsize=(7, 4))
    cols = ["model", "same weekday last week", "7-day mean at origin"]
    horizons.set_index("horizon")[cols].plot.bar(ax=ax, rot=0)
    ax.set(title="Test MAE by forecast horizon", xlabel="days ahead", ylabel="MAE (units)")
    fig.savefig("images/horizons.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
