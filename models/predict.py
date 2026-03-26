import torch
import numpy as np
import pandas as pd
import joblib
import os
import sys
from datetime import timedelta

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from models.lstm_model import StockLSTM, FEATURES, CONTEXT_LEN, PRED_LEN
from data.data_pipeline import fetch_data, add_indicators

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


# load model and scaler
def load_model(ticker: str):
    out_dir = os.path.join(OUTPUTS_DIR, ticker.lower())
    model_path  = os.path.join(out_dir, f"{ticker.lower()}_child_model.pt")
    scaler_path = os.path.join(out_dir, f"{ticker.lower()}_scaler.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No trained model found for {ticker}. Train it first.")

    model = StockLSTM(input_size=len(FEATURES))
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    scaler = joblib.load(scaler_path)
    print(f" Loaded model & scaler for {ticker}")
    return model, scaler


#preparing latest context from stocks
def prepare_context(ticker: str, scaler):
    #last 3 months to ensure 60 days after indicators)
    df = fetch_data(ticker, start=(pd.Timestamp.today() - pd.DateOffset(months=6)).strftime("%Y-%m-%d"))
    df = add_indicators(df)

    if len(df) < CONTEXT_LEN:
        raise ValueError(f"Not enough data for {ticker}. Got {len(df)} rows, need {CONTEXT_LEN}.")

    # Take last 60 days
    context = df[FEATURES].values[-CONTEXT_LEN:]

    # Scale
    context_scaled = scaler.transform(context)
    context_tensor = torch.tensor(context_scaled, dtype=torch.float32).unsqueeze(0)
    # shape: (1, 60, 7)

    return context_tensor, df


#inverse scale
def inverse_scale_close(scaled_values, scaler):
    """Inverse transform only the Close column."""
    close_idx = FEATURES.index("Close")
    n_features = len(FEATURES)

    dummy = np.zeros((len(scaled_values), n_features))
    dummy[:, close_idx] = scaled_values
    inv = scaler.inverse_transform(dummy)[:, close_idx]
    return inv



def predict(ticker: str):
    model, scaler = load_model(ticker)
    context_tensor, df = prepare_context(ticker, scaler)

    with torch.no_grad():
        raw_preds = model(context_tensor).numpy().flatten()
    # raw_preds shape: (5,) — scaled Close prices

    # inverse scale
    prices = inverse_scale_close(raw_preds, scaler)

    # generating forecast dates (skip weekends)
    last_date = df.index[-1]
    forecast_dates = []
    current = last_date
    while len(forecast_dates) < PRED_LEN:
        current += timedelta(days=1)
        if current.weekday() < 5:   # mon–fri only
            forecast_dates.append(current)

    # now build result
    forecast = [
        {"date": d.strftime("%Y-%m-%d"), "predicted_close": round(float(p), 2)}
        for d, p in zip(forecast_dates, prices)
    ]

    return {
        "ticker": ticker,
        "last_known_date": last_date.strftime("%Y-%m-%d"),
        "last_known_close": round(float(df["Close"].iloc[-1]), 2),
        "forecast": forecast
    }

def print_forecast(result: dict):
    print(f"\n{'='*45}")
    print(f"  StockMind Forecast — {result['ticker']}")
    print(f"{'='*45}")
    print(f"  Last known close ({result['last_known_date']}): ${result['last_known_close']}")
    print(f"\n  Next {PRED_LEN}-day forecast:")
    for row in result["forecast"]:
        print(f"    {row['date']}  →  ${row['predicted_close']}")
    print(f"{'='*45}\n")

if __name__ == "__main__":
    result = predict("AAPL")
    print_forecast(result)