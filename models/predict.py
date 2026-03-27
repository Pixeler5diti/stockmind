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

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def load_model(ticker: str):
    out_dir     = os.path.join(OUTPUTS_DIR, ticker.lower())
    model_path  = os.path.join(out_dir, f"{ticker.lower()}_child_model.pt")
    scaler_path = os.path.join(out_dir, f"{ticker.lower()}_scaler.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No trained model found for {ticker}. Train it first.")

    model = StockLSTM(input_size=len(FEATURES))
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    scaler = joblib.load(scaler_path)
    print(f"[+] Loaded model & scaler for {ticker}")
    return model, scaler


def prepare_context(ticker: str, scaler):
    df = fetch_data(
        ticker,
        start=(pd.Timestamp.today() - pd.DateOffset(months=6)).strftime("%Y-%m-%d")
    )
    df = add_indicators(df)

    if len(df) < CONTEXT_LEN:
        raise ValueError(f"Not enough data: got {len(df)}, need {CONTEXT_LEN}")

    context        = df[FEATURES].values[-CONTEXT_LEN:]
    context_scaled = scaler.transform(context)
    context_tensor = torch.tensor(context_scaled, dtype=torch.float32).unsqueeze(0)
    return context_tensor, df


def predict(ticker: str):
    model, scaler   = load_model(ticker)
    context_tensor, df = prepare_context(ticker, scaler)

    with torch.no_grad():
        log_returns = model(context_tensor).numpy().flatten()
    # log_returns shape: (PRED_LEN,) bounded to ±0.05 by tanh in model

    last_close = float(df["Close"].iloc[-1])

    # FIX 1: Reconstruct prices from log returns
    # price_t = price_{t-1} * exp(log_return_t)
    prices = []
    current_price = last_close
    for r in log_returns:
        current_price = current_price * np.exp(r)
        prices.append(current_price)

    # FIX 7: Output price range (±1 std of recent daily moves) instead of fake precision
    recent_returns = np.log(df["Close"] / df["Close"].shift(1)).dropna().values[-30:]
    daily_std      = np.std(recent_returns)

    # Generate forecast dates (skip weekends)
    last_date      = df.index[-1]
    forecast_dates = []
    current        = last_date
    while len(forecast_dates) < PRED_LEN:
        current += timedelta(days=1)
        if current.weekday() < 5:
            forecast_dates.append(current)

    forecast = []
    for i, (d, p, r) in enumerate(zip(forecast_dates, prices, log_returns)):
        uncertainty = daily_std * np.sqrt(i + 1) * p  # grows with horizon
        forecast.append({
            "date":            d.strftime("%Y-%m-%d"),
            "predicted_close": round(float(p), 2),
            "low_estimate":    round(float(p - uncertainty), 2),
            "high_estimate":   round(float(p + uncertainty), 2),
            "pct_change":      round(float(r * 100), 3),  # daily % change
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
    print(f"  StckMind Forecast — {result['ticker']}")
    print(f"{'='*60}")
    print(f"  Last close ({result['last_known_date']}): ${result['last_known_close']}")
    print(f"  30-day daily volatility: {result['daily_volatility']}%")
    print(f"\n  {'Date':<12} {'Change':>8}  {'Low':>9} {'Mid':>9} {'High':>9}")
    print(f"  {'-'*52}")
    for row in result["forecast"]:
        sign = "+" if row["pct_change"] >= 0 else ""
        print(f"  {row['date']:<12} {sign}{row['pct_change']:>7.3f}%  "
              f"${row['low_estimate']:>8.2f} ${row['predicted_close']:>8.2f} ${row['high_estimate']:>8.2f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    result = predict("AAPL")
    print_forecast(result)