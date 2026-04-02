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

OUTPUTS_DIR    = os.path.join(os.path.dirname(__file__), "..", "outputs")
TARGET_VOL     = 0.01
MAX_POSITION   = 1.0
CONF_LONG      = 0.65
CONF_SHORT     = 0.35


def load_model(ticker: str):
    out_dir          = os.path.join(OUTPUTS_DIR, ticker.lower())
    model_path       = os.path.join(out_dir, f"{ticker.lower()}_child_model.pt")
    feat_scaler_path = os.path.join(out_dir, f"{ticker.lower()}_feat_scaler.pkl")
    vol_scaler_path  = os.path.join(out_dir, f"{ticker.lower()}_vol_scaler.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No trained model for {ticker}. Run train.py first.")

    model = StockLSTM(input_size=len(FEATURES))
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    feat_scaler = joblib.load(feat_scaler_path)
    vol_scaler  = joblib.load(vol_scaler_path)

    print(f"[+] Loaded model & scalers for {ticker}")
    return model, feat_scaler, vol_scaler


def prepare_context(ticker: str, feat_scaler):
    df = fetch_data(
        ticker,
        start=(pd.Timestamp.today() - pd.DateOffset(months=6)).strftime("%Y-%m-%d")
    )
    df = add_indicators(df)
    df = add_targets(df)

    if len(df) < CONTEXT_LEN:
        raise ValueError(f"Not enough rows: {len(df)} < {CONTEXT_LEN}")

    context        = df[FEATURES].values[-CONTEXT_LEN:]
    context_scaled = feat_scaler.transform(context)
    context_tensor = torch.tensor(context_scaled, dtype=torch.float32).unsqueeze(0)
    return context_tensor, df


def compute_position(regime_prob: float, recent_vol: float) -> float:
    if regime_prob > CONF_LONG:
        direction = 1.0
    elif regime_prob < CONF_SHORT:
        direction = -1.0
    else:
        return 0.0

    raw_size = TARGET_VOL / (recent_vol + 1e-8)
    position = direction * min(raw_size, MAX_POSITION)
    return round(position, 4)


def predict(ticker: str) -> dict:
    model, feat_scaler, vol_scaler = load_model(ticker)
    context_tensor, df             = prepare_context(ticker, feat_scaler)

    with torch.no_grad():
        pred_vol_scaled, pred_regime = model(context_tensor)

    regime_prob = float(pred_regime.item())
    last_close  = float(df["Close"].iloc[-1])
    last_date   = df.index[-1]

    log_ret    = np.log(df["Close"] / df["Close"].shift(1)).dropna()
    recent_vol = float(log_ret.iloc[-20:].std())
    position   = compute_position(regime_prob, recent_vol)

    if position > 0:
        stance = "LONG"
    elif position < 0:
        stance = "SHORT"
    else:
        stance = "FLAT"

    forecast_dates = []
    current        = last_date
    while len(forecast_dates) < 5:
        current += timedelta(days=1)
        if current.weekday() < 5:
            forecast_dates.append(current)

    forecast = []
    for i, d in enumerate(forecast_dates):
        horizon_vol = recent_vol * np.sqrt(i + 1)
        low         = round(last_close * np.exp(-1.96 * horizon_vol), 2)
        high        = round(last_close * np.exp(+1.96 * horizon_vol), 2)
        forecast.append({
            "date":    d.strftime("%Y-%m-%d"),
            "low_95":  low,
            "high_95": high,
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


def print_forecast(result: dict):
    print(f"\n{'='*60}")
    print(f"  StckMind Signal — {result['ticker']}")
    print(f"{'='*60}")
    print(f"  Last close    : ${result['last_known_close']}  ({result['last_known_date']})")
    print(f"  Regime prob   : {result['regime_prob']:.4f}  (>0.65=long, <0.35=short)")
    print(f"  Stance        : {result['stance']}")
    print(f"  Position size : {result['position_size']:.2%}  of portfolio")
    print(f"  Daily vol     : {result['daily_volatility']}%")
    print(f"\n  5-Day 95% Price Range:")
    print(f"  {'Date':<12}  {'Low':>9}  {'Mid':>9}  {'High':>9}")
    print(f"  {'-'*44}")
    for row in result["forecast"]:
        print(f"  {row['date']:<12}  ${row['low_95']:>8}  ${row['mid']:>8}  ${row['high_95']:>8}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    result = predict("AAPL")
    print_forecast(result)
