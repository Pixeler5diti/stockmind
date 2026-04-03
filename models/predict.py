import torch
import numpy as np
import pandas as pd
import joblib
import os
import sys
from datetime import timedelta

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models.lstm_model import StockLSTM, FEATURES, CONTEXT_LEN
from data.data_pipeline import fetch_data, add_indicators, add_targets

OUTPUTS_DIR  = os.path.join(os.path.dirname(__file__), "..", "outputs")
TARGET_VOL   = 0.01
MAX_POSITION = 1.0
CONF_LONG    = 0.65
CONF_SHORT   = 0.35


def load_model(ticker):
    out_dir          = os.path.join(OUTPUTS_DIR, ticker.lower())
    model_path       = os.path.join(out_dir, f"{ticker.lower()}_child_model.pt")
    feat_scaler_path = os.path.join(out_dir, f"{ticker.lower()}_feat_scaler.pkl")
    vol_scaler_path  = os.path.join(out_dir, f"{ticker.lower()}_vol_scaler.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No trained model for {ticker}. Run train.py first.")
    model = StockLSTM(input_size=len(FEATURES))
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model, joblib.load(feat_scaler_path), joblib.load(vol_scaler_path)


def prepare_context(ticker, feat_scaler):
    df = fetch_data(ticker, start=(pd.Timestamp.today() - pd.DateOffset(months=6)).strftime("%Y-%m-%d"))
    df = add_indicators(df)
    df = add_targets(df)
    if len(df) < CONTEXT_LEN:
        raise ValueError(f"Not enough rows: {len(df)} < {CONTEXT_LEN}")
    context        = df[FEATURES].values[-CONTEXT_LEN:]
    context_scaled = feat_scaler.transform(context)
    context_tensor = torch.tensor(context_scaled, dtype=torch.float32).unsqueeze(0)
    return context_tensor, df


def compute_position(regime_prob, recent_vol):
    if regime_prob > CONF_LONG:
        direction = 1.0
    elif regime_prob < CONF_SHORT:
        direction = -1.0
    else:
        return 0.0
    return round(direction * min(TARGET_VOL / (recent_vol + 1e-8), MAX_POSITION), 4)


def predict(ticker):
    model, feat_scaler, vol_scaler = load_model(ticker)
    context_tensor, df = prepare_context(ticker, feat_scaler)
    with torch.no_grad():
        _, pred_regime = model(context_tensor)
    regime_prob = float(pred_regime.item())
    last_close  = float(df["Close"].iloc[-1])
    last_date   = df.index[-1]
    log_ret     = np.log(df["Close"] / df["Close"].shift(1)).dropna()
    recent_vol  = float(log_ret.iloc[-20:].std())
    position    = compute_position(regime_prob, recent_vol)
    stance      = "LONG" if position > 0 else "SHORT" if position < 0 else "FLAT"
    forecast_dates = []
    current = last_date
    while len(forecast_dates) < 5:
        current += timedelta(days=1)
        if current.weekday() < 5:
            forecast_dates.append(current)
    forecast = []
    for i, d in enumerate(forecast_dates):
        hv   = recent_vol * np.sqrt(i + 1)
        forecast.append({
            "date":    d.strftime("%Y-%m-%d"),
            "low_95":  round(last_close * np.exp(-1.96 * hv), 2),
            "high_95": round(last_close * np.exp(+1.96 * hv), 2),
            "mid":     round(last_close, 2),
        })
    return {
        "ticker":           ticker,
        "last_known_date":  last_date.strftime("%Y-%m-%d"),
        "last_known_close": round(last_close, 2),
        "regime_prob":      round(regime_prob, 4),
        "stance":           stance,
        "position_size":    abs(position),
        "daily_volatility": round(recent_vol * 100, 3),
        "forecast":         forecast,
    }


if __name__ == "__main__":
    r = predict("AAPL")
    print(r)
