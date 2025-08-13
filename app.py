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
import traceback

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
    dummy = np.zeros((1, len(FEATURES)), dtype=float)
    dummy[0, 0] = float(scaled_close_scalar)  # Close is 1st column by design
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

# ---------- Backtest helpers (robust) ----------
def _coerce_numeric_col(df: pd.DataFrame, colname: str) -> None:
    """Ensure df[colname] is a single numeric Series (handles duplicate labels)."""
    obj = df[colname]
    s = obj.iloc[:, 0] if isinstance(obj, pd.DataFrame) else obj
    df[colname] = pd.to_numeric(s, errors="coerce")

def fetch_for_backtest(rows_needed: int, extra_buffer_days: int = 60):
    """Fetch enough OHLCV rows; handle yfinance MultiIndex/dup labels."""
    start = (datetime.utcnow() - timedelta(days=(rows_needed + extra_buffer_days) * 2)).strftime("%Y-%m-%d")
    end = (datetime.utcnow() + timedelta(days=1)).strftime("%Y-%m-%d")
    df = yf.download(TICKER, start=start, end=end)
    if df.empty:
        raise RuntimeError("Could not fetch data from Yahoo Finance for backtest.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(c[0]) if isinstance(c, tuple) else str(c) for c in df.columns]
    else:
        df.columns = [str(c) for c in df.columns]

    df = df.reset_index().rename(columns={"Date": "date"}).sort_values("date")
    needed_lower = ["date", "open", "high", "low", "close", "volume"]
    have_lower = [c.lower() for c in df.columns]
    missing = [n for n in needed_lower if n not in have_lower]
    if missing:
        raise RuntimeError(f"Missing expected columns from Yahoo Finance: {missing}")

    df = df[["date", "Open", "High", "Low", "Close", "Volume"]].copy()
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        _coerce_numeric_col(df, c)

    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    if len(df) < rows_needed:
        raise RuntimeError(f"Not enough trading rows fetched ({len(df)} < {rows_needed}). Try increasing buffer.")
    return df.tail(rows_needed).copy()

def backtest_last_n_days(model, scaler, horizon: int = 7, lookback: int = 60):
    """Rolling backtest for the last `horizon` trading days."""
    rows_needed = lookback + horizon
    df_tail = fetch_for_backtest(rows_needed)

    feats = ["Close", "High", "Low", "Open", "Volume"]
    rows = []

    for i in range(lookback, lookback + horizon):
        window_df = df_tail.iloc[i - lookback : i][feats].copy()
        if window_df.shape != (lookback, len(feats)):
            raise RuntimeError(f"Window shape bad: {window_df.shape}, expected {(lookback, len(feats))}")
        window_df = window_df.astype(float)

        actual_close = float(df_tail.iloc[i]["Close"])
        last_close   = float(df_tail.iloc[i - 1]["Close"])
        pred_for_date = pd.to_datetime(df_tail.iloc[i]["date"])

        x_scaled_3d = scaler.transform(window_df).astype(float).reshape(1, lookback, len(feats))
        scaled_pred = model.predict(x_scaled_3d, verbose=0)
        scaled_pred_scalar = float(np.asarray(scaled_pred).ravel()[0])
        predicted_close = inverse_close_from_scalar(scaler, scaled_pred_scalar)

        direction_pred = "UP" if predicted_close > last_close else "DOWN"
        direction_true = "UP" if actual_close > last_close else "DOWN"
        direction_hit  = bool(direction_pred == direction_true)

        abs_err = float(abs(predicted_close - actual_close))
        pct_err = float(abs_err / actual_close) if actual_close != 0 else np.nan

        rows.append({
            "prediction_for": pred_for_date,
            "last_close": float(last_close),
            "predicted_close": float(predicted_close),
            "actual_close": float(actual_close),
            "direction_pred": direction_pred,
            "direction_true": direction_true,
            "direction_hit": direction_hit,
            "abs_error": abs_err,
            "pct_error": pct_err,
        })

    res = pd.DataFrame(rows).sort_values("prediction_for").reset_index(drop=True)

    mae  = float(np.mean(res["abs_error"]))
    rmse = float(np.sqrt(np.mean((res["predicted_close"] - res["actual_close"])**2)))
    mape = float(np.mean(res["pct_error"])) * 100.0
    da   = float(res["direction_hit"].mean()) * 100.0

    metrics = {"MAE": mae, "RMSE": rmse, "MAPE_%": mape, "Directional_Accuracy_%": da}
    return res, metrics

# ---------- Trade/No-Trade + costs + rolling metrics ----------
def decide_signal(pred_close: float, last_close: float, band_half: float, k: float = 1.0) -> str:
    """NO TRADE if |predicted move| ≤ k * band_half (expected error proxy)."""
    delta = abs(float(pred_close) - float(last_close))
    if delta <= float(k) * float(band_half):
        return "NO TRADE"
    return "LONG" if pred_close > last_close else "SHORT"

def add_costs_and_filter_pl(df: pd.DataFrame, per_side_cost_pct: float = 0.001, expected_err_window: int = 5) -> pd.DataFrame:
    """
    Use rolling expected error (from prior abs errors) to decide NO TRADE.
    Apply per-side cost to executed trades. Returns pl/enriched DF.
    """
    out = df.copy().sort_values("prediction_for").reset_index(drop=True)

    # Expected error = mean of previous abs errors; fallback ~0.8% of price
    fallback = out["actual_close"].astype(float) * 0.008
    out["expected_err"] = (
        out["abs_error"].shift(1)
          .rolling(expected_err_window, min_periods=1)
          .mean()
          .fillna(fallback)
    )

    # Decide signal using expected_err as band_half proxy
    signals, trades = [], []
    for _, r in out.iterrows():
        sig = decide_signal(r["predicted_close"], r["last_close"], r["expected_err"], k=1.0)
        signals.append(sig)
        trades.append(sig != "NO TRADE")
    out["signal"] = signals
    out["trade_executed"] = trades

    pos = out["signal"].map({"LONG": 1, "SHORT": -1, "NO TRADE": 0}).astype(int)

    out["point_move"] = (out["actual_close"].astype(float) - out["last_close"].astype(float))
    out["pl_points_gross"] = out["point_move"] * pos
    out["return_pct_gross"] = ((out["actual_close"].astype(float) - out["last_close"].astype(float))
                               / out["last_close"].astype(float)) * pos

    # Round-trip cost (enter + exit)
    round_trip_cost = 2.0 * float(per_side_cost_pct)
    out["return_pct_net"] = np.where(out["trade_executed"], out["return_pct_gross"] - round_trip_cost, 0.0)
    out["pl_points_net"] = np.where(out["trade_executed"],
                                    out["pl_points_gross"] - (out["last_close"].astype(float) * round_trip_cost),
                                    0.0)

    out["cum_pl_points_net"] = out["pl_points_net"].cumsum()
    out["cum_return_pct_net"] = (1.0 + out["return_pct_net"]).cumprod() - 1.0
    return out

def compute_rolling_perf(df: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """Rolling DA% and RMSE with min_periods=1 so lines show from start."""
    out = df.sort_values("prediction_for").reset_index(drop=True).copy()
    out["rolling_DA_%"] = (
        out["direction_hit"].astype(float).rolling(window=window, min_periods=1).mean() * 100.0
    )
    se = (out["predicted_close"].astype(float) - out["actual_close"].astype(float)) ** 2
    out["rolling_RMSE"] = se.rolling(window=window, min_periods=1).mean().pow(0.5)
    out["roll_window"] = int(window)
    return out
# ---------- End helpers ----------

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

        # ---- KPIs + Trade/No-Trade badge ----
        band_half = max(1e-9, (upper - lower) / 2.0)
        signal = decide_signal(predicted_close, last_close, band_half, k=1.0)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Predicted Close (t+1)", f"{predicted_close:,.2f}")
        delta = predicted_close - last_close
        k2.metric("Direction", "↑ UP" if direction=="UP" else "↓ DOWN", f"{delta:,.2f}")
        k3.metric("Confidence Band", f"{lower:,.2f}  →  {upper:,.2f}")
        k4.metric("Signal", signal)

        st.caption(f"Rule: **NO TRADE** if |move| ≤ half‑band. Now: |{delta:,.2f}| vs {band_half:,.2f}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.exception(e)

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
st.download_button("⬇️ Download CSV History", data=csv_buf.getvalue(),
                   file_name="ftse_predictions_history.csv", mime="text/csv")

st.caption("Note: Confidence band uses a rolling RMSE from your recent finalized predictions; bands tighten as history grows.")

# ---------------------------
# Backtest (with P/L, costs & rolling charts)
# ---------------------------
st.markdown("---")
st.subheader("Backtest (recent trading days)")

col_bt1, col_bt2 = st.columns([1, 2])

with col_bt1:
    horizon = st.number_input(
        "Backtest horizon (trading days)",
        min_value=3, max_value=60, value=7, step=1,
        help="How many recent trading days to evaluate."
    )
    roll_window = st.number_input(
        "Rolling window (days)",
        min_value=3, max_value=30, value=min(7, int(horizon)), step=1,
        help="Window size for rolling DA/RMSE charts."
    )
    cost = st.number_input("Per-side cost (e.g., 0.10% = 0.001)",
                           min_value=0.0, max_value=0.01, value=0.001, step=0.0005)
    exp_win = st.number_input("Expected error window (days)",
                              min_value=3, max_value=20, value=5, step=1)
    run_bt = st.button("▶️ Run Backtest", use_container_width=True)

if run_bt:
    try:
        with st.spinner(f"Running backtest for last {int(horizon)} trading days..."):
            bt_df, bt_metrics = backtest_last_n_days(model, scaler, horizon=int(horizon), lookback=LOOKBACK)

        # ------- KPIs (model) -------
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("MAE", f"{bt_metrics['MAE']:,.2f}")
        k2.metric("RMSE", f"{bt_metrics['RMSE']:,.2f}")
        k3.metric("MAPE", f"{bt_metrics['MAPE_%']:.2f}%")
        k4.metric("Directional Acc.", f"{bt_metrics['Directional_Accuracy_%']:.1f}%")

        # ------- Benchmark (naïve = yesterday's close) -------
        naive_abs_err = (bt_df["last_close"] - bt_df["actual_close"]).abs()
        naive_rmse = float(np.sqrt(np.mean((bt_df["last_close"] - bt_df["actual_close"])**2)))
        nb1, nb2 = st.columns(2)
        nb1.metric("Naïve MAE", f"{float(naive_abs_err.mean()):,.2f}")
        nb2.metric("Naïve RMSE", f"{naive_rmse:,.2f}")

        # ------- P/L simulation with NO-TRADE filter & costs -------
        pl_df = add_costs_and_filter_pl(bt_df, per_side_cost_pct=float(cost), expected_err_window=int(exp_win))
        hits = int(pl_df["direction_hit"].sum())
        total = int(len(pl_df))
        executed = int(pl_df["trade_executed"].sum())
        st.caption(f"Direction hits: **{hits}/{total}** • Executed trades: **{executed}/{total}** (after filter)")

        import altair as alt

        # Cumulative P/L (net, points)
        pl_points_chart = alt.Chart(pl_df).mark_line(point=True).encode(
            x=alt.X("prediction_for:T", title="Date"),
            y=alt.Y("cum_pl_points_net:Q", title="Cumulative P/L (net, points)")
        ).properties(height=260)
        st.altair_chart(pl_points_chart, use_container_width=True)

        # Cumulative return (net, %)
        cum_ret_chart = alt.Chart(pl_df).mark_line(point=True).encode(
            x=alt.X("prediction_for:T", title="Date"),
            y=alt.Y("cum_return_pct_net:Q", title="Cumulative Return (net, %)", axis=alt.Axis(format=".1%"))
        ).properties(height=260)
        st.altair_chart(cum_ret_chart, use_container_width=True)

        # ------- Rolling performance -------
        roll_df = compute_rolling_perf(bt_df, window=int(roll_window))

        roll_da_chart = alt.Chart(roll_df).mark_line(point=True).encode(
            x=alt.X("prediction_for:T", title="Date"),
            y=alt.Y("rolling_DA_%:Q", title="Rolling Directional Accuracy (%)")
        ).properties(height=260)

        roll_rmse_chart = alt.Chart(roll_df).mark_line(point=True).encode(
            x=alt.X("prediction_for:T", title="Date"),
            y=alt.Y("rolling_RMSE:Q", title="Rolling RMSE")
        ).properties(height=260)

        st.altair_chart(roll_da_chart, use_container_width=True)
        st.altair_chart(roll_rmse_chart, use_container_width=True)

        # ------- Table (pretty % formatting) -------
        table_df = pl_df.copy()
        table_df["pct_error"] = (table_df["pct_error"] * 100).round(3)
        table_df["return_pct_gross"] = (table_df["return_pct_gross"] * 100).round(3)
        table_df["return_pct_net"] = (table_df["return_pct_net"] * 100).round(3)
        table_df["cum_return_pct_net"] = (table_df["cum_return_pct_net"] * 100).round(3)
        st.dataframe(table_df.reset_index(drop=True), use_container_width=True)

        # CSV download
        out = io.StringIO()
        pl_df.to_csv(out, index=False)
        st.download_button(
            "⬇️ Download Backtest CSV",
            data=out.getvalue(),
            file_name=f"backtest_last_{int(horizon)}_trading_days.csv",
            mime="text/csv"
        )

        st.caption("Backtest uses only the prior 60 trading days for each target date (no look‑ahead).")

    except Exception as e:
        st.error(f"Backtest failed: {e}")
        st.exception(e)
