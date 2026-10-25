# Demo of the retail forecaster. Run train.py first, then: streamlit run app.py

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from train import FEATURES, MODEL_FILE, add_features

st.set_page_config(page_title="Retail demand forecast", layout="wide")
st.title("Retail demand forecast")
st.write("Replays the 1-day-ahead forecast of total units sold on days in the test period (UCI Online Retail II).")

if not MODEL_FILE.exists() or not Path("data/test_predictions.csv").exists():
    st.error("Model not found. Run python train.py first.")
    st.stop()


@st.cache_data
def load_data():
    daily = pd.read_csv("data/daily.csv", index_col=0, parse_dates=True)
    preds = pd.read_csv("data/test_predictions.csv", index_col=0, parse_dates=True)
    metrics = pd.read_csv("models/metrics.csv")
    return add_features(daily), preds, metrics


@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


daily, preds, metrics = load_data()
saved = load_model()

days = list(preds.index.date)
day = st.sidebar.selectbox("Test day", days, index=len(days) - 1)
history = st.sidebar.slider("Days of history to show", 14, 90, 28)
day = pd.Timestamp(day)
row = preds.loc[day]

# MAE of the saved model and of the "same day last week" baseline, from train.py output
model_name = "GB, log target" if saved["log_target"] else "GB, raw target"
mae = metrics.set_index("method")["MAE"]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Actual units", f"{row['actual']:,.0f}")
col2.metric("Forecast", f"{row['predicted']:,.0f}", f"{row['predicted'] - row['actual']:+,.0f} vs actual",
            delta_color="off")
col3.metric("Model MAE (test)", f"{mae[model_name]:,.0f}")
col4.metric("Baseline MAE (same day last week)", f"{mae['Same day last week']:,.0f}")
inside = row["lower"] <= row["actual"] <= row["upper"]
st.write(f"10-90% range: {row['lower']:,.0f} to {row['upper']:,.0f} units. "
         f"Actual is {'inside' if inside else 'outside'} the range.")

# Recent history up to the chosen day, with the forecast and range for test days
start = day - pd.Timedelta(days=history)
recent = daily.loc[start:day, ["units"]]
test_part = preds.loc[start:day]

fig, ax = plt.subplots(figsize=(11, 3.5))
ax.plot(recent.index, recent["units"], ".-", label="actual units (0 = closed)")
ax.fill_between(test_part.index, test_part["lower"], test_part["upper"], alpha=0.2, color="tab:orange",
                label="10-90% range")
ax.plot(test_part.index, test_part["predicted"], ".--", color="tab:orange", label="forecast")
ax.plot([day], [row["predicted"]], "o", color="red", markersize=9, label="forecast for chosen day")
ax.set_ylabel("units")
ax.legend(loc="upper left")
fig.autofmt_xdate()
st.pyplot(fig)
plt.close(fig)

st.subheader("Last 10 open days up to the chosen day")
table = test_part[["actual", "predicted", "lower", "upper", "baseline"]].tail(10)
table.index = table.index.date
st.dataframe(table.astype(int), width="stretch")

with st.expander("How it works"):
    st.write(
        "The model is a HistGradientBoostingRegressor trained on daily total units up to 2011-09-08. "
        "For each day it only sees earlier days: units 1, 7 and 14 days ago, the mean of the previous "
        "7 and 28 days, the day of week and the month. Two quantile models give the 10-90% range. "
        "The baseline simply repeats the same weekday last week. Closed days are not scored."
    )
    st.write("Features the model saw for the chosen day:")
    st.dataframe(daily.loc[[day], FEATURES], hide_index=True)
