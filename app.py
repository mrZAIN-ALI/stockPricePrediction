import os
import io
import time
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model

# ---------------------------
# App config
# ---------------------------
st.set_page_config(page_title="FTSE-100 Forecast", page_icon="📈", layout="wide")

TICKER = "^FTSE"               # FTSE 100 index
LOOKBACK = 60                  # last 60 trading days
FEATURES = ["Close", "High", "Low", "Open", "Volume"]
MODEL_PATH = "best_lstm_model.h5"
SCALER_PATH = "scaler.save"

# ---------------------------
# Caching: model & scaler
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_assets():
    model = load_model(MODEL_PATH, compile=False)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

# ---------------------------
# Helpers
# ---------------------------
def fetch_ohlcv_last_60():
    # Download a little extra and then take the last 60 trading rows
    start = (datetime.utcnow() - timedelta(days=180)).strftime("%Y-%m-%d")
    end = (datetime.utcnow() + timedelta(days=1)).strftime("%Y-%m-%d")
    df = yf.download(TICKER, start=start, end=end)
    if df.empty:
        raise RuntimeError("Could not fetch data from Yahoo Finance.")
    # Normalize column names to expected set and order
    df = df.reset_index().rename(columns={"Date": "date"})
    want = ["date", "Open", "High", "Low", "Close", "Volume"]
    df = df[want].dropna()
    df = df.sort_values("date")
    # Take the last 60 trading rows, with FEATURES order
    last_60 = df.tail(LOOKBACK).copy()
    X60 = last_60[["Close", "High", "Low", "Open", "Volume"]].copy()
    return df, last_60, X60

def inverse_close_from_scalar(scaler, scaled_close_scalar):
    """Place predicted scaled close into a 1x5 dummy row to inverse-transform."""
    dummy = np.zeros((1, len(FEATURES)))
    dummy[0, 0] = scaled_close_scalar  # Close is 1st column by design
    inv = scaler.inverse_transform(dummy)
    return float(inv[0, 0])

def next_business_day(d):
    # Simple next business day (Mon-Fri). LSE holidays not considered in v1.
    wd = d.weekday()
    if wd >= 4:  # Fri=4, Sat=5, Sun=6
        return d + timedelta(days=(7 - wd))
    return d + timedelta(days=1)

def compute_confidence_band(history_df, predicted_close):
    # Use rolling RMSE from last N finalized rows (with actuals)
    N = 30
    if history_df is None or history_df.empty:
        return predicted_close, predicted_close
    finalized = history_df.dropna(subset=["actual_close"])
    if len(finalized) < 5:
        # Fallback: naive +/- 0.8% band if we don't yet have enough finalized rows
        band = predicted_close * 0.008
        return predicted_close - band, predicted_close + band
    lastN = finalized.tail(N)
    rmse = np.sqrt(np.mean((lastN["predicted_close"] - lastN["actual_close"])**2))
    return predicted_close - rmse, predicted_close + rmse

def reconcile_latest_actuals(history_df):
    """Try to fill actual_close for any predictions whose date has passed."""
    if history_df is None or history_df.empty:
        return history_df
    # Pull recent actuals for mapping
    start = (datetime.utcnow() - timedelta(days=365)).strftime("%Y-%m-%d")
    end = (datetime.utcnow() + timedelta(days=1)).strftime("%Y-%m-%d")
    df = yf.download(TICKER, start=start, end=end)
    if df.empty:
        return history_df
    df = df.reset_index().rename(columns={"Date": "date"}).sort_values("date")
    close_map = dict(zip(pd.to_datetime(df["date"]).dt.date, df["Close"].astype(float)))

    updated = history_df.copy()
    for i, row in updated.iterrows():
        pred_for = pd.to_datetime(row["prediction_for"]).date()
        if pd.isna(row["actual_close"]) and pred_for in close_map:
            updated.at[i, "actual_close"] = close_map[pred_for]
            # compute errors
            if not pd.isna(updated.at[i, "actual_close"]):
                err = abs(updated.at[i, "predicted_close"] - updated.at[i, "actual_close"])
                updated.at[i, "abs_error"] = err
                if updated.at[i, "actual_close"] != 0:
                    updated.at[i, "pct_error"] = err / updated.at[i, "actual_close"]
                # direction hit
                direction_pred = updated.at[i, "direction_pred"]
                last_close = updated.at[i, "last_close"]
                if not pd.isna(last_close):
                    true_dir = "UP" if updated.at[i, "actual_close"] > last_close else "DOWN"
                    updated.at[i, "direction_hit"] = (direction_pred == true_dir)
    return updated

def init_history():
    cols = [
        "generated_at", "window_start", "window_end", "prediction_for",
        "last_close", "predicted_close", "actual_close",
        "direction_pred", "direction_hit", "abs_error", "pct_error",
        "model_version", "scaler_version"
    ]
    return pd.DataFrame(columns=cols)

# ---------------------------
# State
# ---------------------------
if "history" not in st.session_state:
    st.session_state["history"] = init_history()

# ---------------------------
# UI
# ---------------------------
st.title("📈 FTSE‑100 Next‑Day Forecast")
st.caption("LSTM (600‑fold walk‑forward, 2015–2025) • Minimal, modern Streamlit demo")

colA, colB = st.columns([1, 1])

with colA:
    st.subheader("Market Snapshot (last 60 trading days)")
with colB:
    st.subheader("Prediction")

# Load model & scaler
with st.spinner("Loading model and scaler..."):
    model, scaler = load_assets()

# Data section
try:
    full_df, last_60_df, X60 = fetch_ohlcv_last_60()
except Exception as e:
    st.error(f"Data fetch failed: {e}")
    st.stop()

# Draw price chart
import altair as alt
price_chart = alt.Chart(last_60_df).mark_line().encode(
    x=alt.X('date:T', title='Date'),
    y=alt.Y('Close:Q', title='Close')
).properties(height=280)
st.altair_chart(price_chart, use_container_width=True)

# Actions
predict_btn = st.button("🔮 Predict Next Close", use_container_width=True)
reconcile_btn = st.button("↺ Reconcile Actuals", help="Fill actual closes for past predictions", use_container_width=True)

# Reconcile actuals if requested
if reconcile_btn:
    with st.spinner("Reconciling with latest actuals..."):
        st.session_state["history"] = reconcile_latest_actuals(st.session_state["history"])
        st.success("Reconciled (where data was available).")

# Prediction
if predict_btn:
    try:
        # Prepare input
        x_scaled = scaler.transform(X60[FEATURES]).reshape(1, LOOKBACK, len(FEATURES))
        # Inference
        with st.spinner("Running model inference..."):
            scaled_pred = model.predict(x_scaled, verbose=0)
        predicted_close = inverse_close_from_scalar(scaler, float(scaled_pred.ravel()[0]))
        last_close = float(last_60_df["Close"].iloc[-1])

        direction = "UP" if predicted_close > last_close else "DOWN"
        lower, upper = compute_confidence_band(st.session_state["history"], predicted_close)

        # Dates
        window_start = pd.to_datetime(last_60_df["date"].iloc[0])
        window_end = pd.to_datetime(last_60_df["date"].iloc[-1])
        prediction_for = next_business_day(window_end.to_pydatetime())

        # Append to history (actual will be filled later)
        row = {
            "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "window_start": window_start,
            "window_end": window_end,
            "prediction_for": prediction_for,
            "last_close": last_close,
            "predicted_close": predicted_close,
            "actual_close": np.nan,
            "direction_pred": direction,
            "direction_hit": np.nan,
            "abs_error": np.nan,
            "pct_error": np.nan,
            "model_version": "lstm-600f-v1",
            "scaler_version": "minmax-v1",
        }
        st.session_state["history"] = pd.concat([st.session_state["history"], pd.DataFrame([row])], ignore_index=True)

        # KPIs
        k1, k2, k3 = st.columns(3)
        k1.metric("Predicted Close (t+1)", f"{predicted_close:,.2f}")
        delta = predicted_close - last_close
        k2.metric("Direction", "↑ UP" if direction=="UP" else "↓ DOWN", f"{delta:,.2f}")
        k3.metric("Confidence Band", f"{lower:,.2f}  →  {upper:,.2f}")

        # Plot predicted point
        pred_df = pd.DataFrame({"date": [prediction_for], "Predicted": [predicted_close]})
        pred_chart = alt.Chart(pred_df).mark_point(size=80, filled=True).encode(
            x='date:T', y='Predicted:Q'
        ).properties(height=0)  # overlay not needed; just echo values below
        st.caption(f"Next trading day: **{prediction_for.date()}** • Band: **[{lower:,.2f}, {upper:,.2f}]**")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.markdown("---")

# History + CSV
st.subheader("Predictions vs Actuals (recent)")
history = st.session_state["history"].copy()

# Compute rolling metrics for display (recent finalized)
finalized = history.dropna(subset=["actual_close"])
if not finalized.empty:
    rolling_window = min(30, len(finalized))
    rmse = np.sqrt(np.mean((finalized.tail(rolling_window)["predicted_close"]
                            - finalized.tail(rolling_window)["actual_close"])**2))
    da = (finalized["direction_hit"].tail(rolling_window).mean()) if "direction_hit" in finalized else np.nan
else:
    rmse, da = np.nan, np.nan

m1, m2 = st.columns(2)
m1.metric("Recent RMSE (finalized)", "—" if pd.isna(rmse) else f"{rmse:,.2f}")
m2.metric("Recent Directional Accuracy", "—" if pd.isna(da) else f"{da*100:.1f}%")

st.dataframe(
    history.sort_values("generated_at", ascending=False),
    use_container_width=True
)

csv_buf = io.StringIO()
history.to_csv(csv_buf, index=False)
st.download_button("⬇️ Download CSV History", data=csv_buf.getvalue(), file_name="ftse_predictions_history.csv", mime="text/csv")

st.caption("Note: Confidence band uses a rolling RMSE from your recent finalized predictions; bands tighten as history grows.")
