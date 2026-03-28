import torch
import numpy as np
import pandas as pd
import joblib
import os
import sys
from datetime import timedelta

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models.lstm_model import StockLSTM, FEATURES, CONTEXT_LEN, PRED_LEN
from data.data_pipeline import fetch_data, add_indicators

OUTPUTS_DIR  = os.path.join(os.path.dirname(__file__), "..", "outputs")
MAX_DAILY_MOVE = 0.05   # ±5% hard cap per day


def load_model(ticker: str):
    out_dir    = os.path.join(OUTPUTS_DIR, ticker.lower())
    model_path = os.path.join(out_dir, f"{ticker.lower()}_child_model.pt")
    scaler_path= os.path.join(out_dir, f"{ticker.lower()}_scaler.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No trained model found for {ticker}.")

    model = StockLSTM(input_size=len(FEATURES))
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    scaler = joblib.load(scaler_path)
    return model, scaler


def prepare_context(ticker: str, scaler):
    df = fetch_data(
        ticker,
        start=(pd.Timestamp.today() - pd.DateOffset(months=6)).strftime("%Y-%m-%d")
    )
    df = add_indicators(df)

    if len(df) < CONTEXT_LEN:
        raise ValueError(f"Not enough rows: {len(df)} < {CONTEXT_LEN}")

    # FIX 3: scale context using the SAME scaler used during training
    context        = df[FEATURES].values[-CONTEXT_LEN:]
    context_scaled = scaler.transform(context)
    context_tensor = torch.tensor(context_scaled, dtype=torch.float32).unsqueeze(0)
    return context_tensor, df


def predict(ticker: str):
    model, scaler      = load_model(ticker)
    context_tensor, df = prepare_context(ticker, scaler)

    with torch.no_grad():
        raw_output = model(context_tensor).numpy().flatten()
    # raw_output is already bounded to ±0.05 by tanh*0.05 in forward()
    # These ARE the log returns directly — no inverse scaling needed
    # because the target (log returns) was never scaled during training

    # FIX 2+5: clamp to ±MAX_DAILY_MOVE as safety net
    log_returns = np.clip(raw_output, -MAX_DAILY_MOVE, MAX_DAILY_MOVE)

    # Sanity check — log returns should be small numbers
    print(f"  [debug] raw log returns: {np.round(log_returns, 5)}")

    last_close = float(df["Close"].iloc[-1])

    # FIX 1+4: step-by-step compounding
    # price_t = price_{t-1} * exp(log_return_t)
    prices = []
    prev   = last_close
    for r in log_returns:
        next_price = prev * np.exp(r)
        prices.append(next_price)
        prev = next_price   # FIX 4: use predicted price as base for next step

    # Recent volatility for range estimates
    recent_log_ret = np.log(
        df["Close"] / df["Close"].shift(1)
    ).dropna().values[-30:]
    daily_std = np.std(recent_log_ret)

    # Skip weekends
    last_date      = df.index[-1]
    forecast_dates = []
    current        = last_date
    while len(forecast_dates) < PRED_LEN:
        current += timedelta(days=1)
        if current.weekday() < 5:
            forecast_dates.append(current)

    forecast = []
    for i, (d, p, r) in enumerate(zip(forecast_dates, prices, log_returns)):
        # Uncertainty grows with horizon (sqrt of time)
        uncertainty = daily_std * np.sqrt(i + 1) * p
        pct_change  = (p - (prices[i-1] if i > 0 else last_close)) / (prices[i-1] if i > 0 else last_close) * 100
        forecast.append({
            "date":            d.strftime("%Y-%m-%d"),
            "predicted_close": round(float(p), 2),
            "low_estimate":    round(float(p - uncertainty), 2),
            "high_estimate":   round(float(p + uncertainty), 2),
            "pct_change":      round(float(pct_change), 3),
        })

    return {
        "ticker":           ticker,
        "last_known_date":  last_date.strftime("%Y-%m-%d"),
        "last_known_close": round(last_close, 2),
        "forecast":         forecast,
        "daily_volatility": round(float(daily_std * 100), 3),
    }


def print_forecast(result: dict):
    print(f"\n{'='*60}")
    print(f"  StckMind — {result['ticker']}")
    print(f"{'='*60}")
    print(f"  Last close  : ${result['last_known_close']}  ({result['last_known_date']})")
    print(f"  30d vol     : {result['daily_volatility']}% / day")
    print(f"\n  {'Date':<12} {'Δ%':>8}   {'Low':>9}  {'Mid':>9}  {'High':>9}")
    print(f"  {'-'*55}")
    for row in result["forecast"]:
        sign = "+" if row["pct_change"] >= 0 else ""
        print(
            f"  {row['date']:<12} {sign}{row['pct_change']:>7.3f}%   "
            f"${row['low_estimate']:>8.2f}  ${row['predicted_close']:>8.2f}  ${row['high_estimate']:>8.2f}"
        )
    print(f"{'='*60}\n")


if __name__ == "__main__":
    result = predict("AAPL")
    print_forecast(result)