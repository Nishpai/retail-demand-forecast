import joblib
import numpy as np
import pandas as pd
import pytest

from train import FEATURES, MODEL_FILE, add_features, clean, make_daily


def test_clean_drops_cancellations_and_postage():
    df = pd.DataFrame({
        "Invoice": ["100", "C101", "102", "103"],
        "StockCode": ["85123A", "85123A", "POST", "22423"],
        "Quantity": [6, -6, 1, 0],
        "Price": [2.5, 2.5, 18.0, 12.0],
        "Customer ID": [1.0, 2.0, 1.0, 1.0],
        "InvoiceDate": pd.to_datetime(["2010-01-04"] * 4),
    })
    assert clean(df)["Invoice"].tolist() == ["100"]


def test_closed_days_are_filled_with_zero():
    df = pd.DataFrame({"Quantity": [5, 7],
                       "InvoiceDate": pd.to_datetime(["2010-01-01 10:00", "2010-01-04 12:00"])})
    assert make_daily(df)["units"].tolist() == [5, 0, 0, 7]


def test_features_only_use_the_past():
    days = pd.date_range("2010-01-01", periods=40, freq="D")
    daily = add_features(pd.DataFrame({"units": np.arange(40.0)}, index=days))
    # Changing today's value must not change any of today's features.
    changed = daily[["units"]].copy()
    changed.iloc[-1, 0] = 1_000_000
    assert add_features(changed).iloc[-1][FEATURES].equals(daily.iloc[-1][FEATURES])
    assert daily.iloc[-1]["last"] == 38 and daily.iloc[-1]["lag_7"] == 32


@pytest.mark.skipif(not MODEL_FILE.exists(), reason="run train.py first")
def test_saved_model_predicts_one_value_per_row():
    saved = joblib.load(MODEL_FILE)
    daily = add_features(pd.read_csv("data/daily.csv", index_col=0, parse_dates=True)).dropna()
    pred = saved["model"].predict(daily[FEATURES].tail(10))
    assert pred.shape == (10,)


def test_cancelled_order_and_its_original_are_both_removed():
    df = pd.DataFrame({
        "Invoice": ["200", "201", "C202"],
        "StockCode": ["23843", "23843", "23843"],
        "Quantity": [80995, 5, -80995],
        "Price": [2.08, 2.08, 2.08],
        "Customer ID": [16446.0, 16446.0, 16446.0],
        "InvoiceDate": pd.to_datetime(["2011-12-09 09:15", "2011-12-09 09:20", "2011-12-09 09:27"]),
    })
    assert clean(df)["Invoice"].tolist() == ["201"]


@pytest.mark.skipif(not MODEL_FILE.exists(), reason="run train.py first")
def test_lower_is_not_above_upper():
    saved = joblib.load(MODEL_FILE)
    daily = add_features(pd.read_csv("data/daily.csv", index_col=0, parse_dates=True)).dropna()
    rows = daily[daily["units"] > 0][FEATURES]
    assert (saved["lower"].predict(rows) <= saved["upper"].predict(rows)).all()


def test_horizon_features_only_use_the_origin_day_or_earlier():
    days = pd.date_range("2010-01-01", periods=60, freq="D")
    daily = pd.DataFrame({"units": np.arange(60.0)}, index=days)
    for h in (1, 3, 7):
        before = add_features(daily, h).iloc[-1][FEATURES]
        # The last h days come after the origin day, so changing them must not matter.
        changed = daily.copy()
        changed.iloc[-h:, 0] = 1_000_000
        assert add_features(changed, h).iloc[-1][FEATURES].equals(before)


def test_last_known_value_is_h_days_back():
    days = pd.date_range("2010-01-01", periods=60, freq="D")
    daily = pd.DataFrame({"units": np.arange(60.0)}, index=days)
    for h in (1, 3, 7):
        row = add_features(daily, h).iloc[-1]
        assert row["last"] == 59 - h
        assert row["mean_7"] == np.mean(range(59 - h - 6, 59 - h + 1))
