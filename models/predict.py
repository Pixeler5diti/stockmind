import torch
import numpy as np
import pandas as pd
import joblib
import os
import sys
from datetime import timedelta

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models.lstm_model import StockLSTM, create_sequences, CONTEXT_LEN
from data.data_pipeline import fetch_data, add_indicators, add_targets

OUTPUTS_DIR  = os.path.join(os.path.dirname(__file__), "..", "outputs")
TARGET_VOL   = 0.01    # 1% daily portfolio vol target
MAX_POSITION = 1.0     # no leverage
CONF_LONG    = 0.65    # sigmoid > this -> LONG
CONF_SHORT   = 0.35    # sigmoid < this -> SHORT


def get_features(df):
    """Mirror the feature selection from train.py."""
    exclude = {"vol_regime", "vol_direction", "forward_return_1d",
               "Open", "High", "Low", "Close", "Volume"}
    return [c for c in df.columns if c not in exclude]


def load_model(ticker: str):
    out_dir     = os.path.join(OUTPUTS_DIR, ticker.lower())
    model_path  = os.path.join(out_dir, f"{ticker.lower()}_child_model.pt")
    scaler_path = os.path.join(out_dir, f"{ticker.lower()}_scaler.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No trained model for {ticker}. Run train.py first.")

    scaler   = joblib.load(scaler_path)
    features = scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else None
    n_features = scaler.n_features_in_

    model = StockLSTM(input_size=n_features)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    print(f"[+] Loaded model ({n_features} features) for {ticker}")
    return model, scaler, n_features


def prepare_context(ticker: str, scaler, n_features: int):
    df = fetch_data(
        ticker,
        start=(pd.Timestamp.today() - pd.DateOffset(months=6)).strftime("%Y-%m-%d")
    )
    df = add_indicators(df)
    df = add_targets(df)
    df = df.replace([float('inf'), float('-inf')], float('nan')).dropna()

    features = get_features(df)

    # Ensure feature count matches what the scaler was trained on
    if len(features) != n_features:
        raise ValueError(
            f"Feature mismatch: model expects {n_features} features, "
            f"got {len(features)}. Re-run data_pipeline.py and train.py."
        )

    if len(df) < CONTEXT_LEN:
        raise ValueError(f"Not enough rows: {len(df)} < {CONTEXT_LEN}")

    context        = df[features].values[-CONTEXT_LEN:]
    context_scaled = scaler.transform(context)
    context_tensor = torch.tensor(context_scaled, dtype=torch.float32).unsqueeze(0)

    return context_tensor, df, features


def compute_position(sigmoid_output: float, recent_vol: float) -> float:
    """
    Volatility-targeted position sizing.
    Direction from regime classifier output (thresholded sigmoid).
    Size = TARGET_VOL / realized_vol, capped at MAX_POSITION.
    """
    if sigmoid_output > CONF_LONG:
        direction = 1.0
    elif sigmoid_output < CONF_SHORT:
        direction = -1.0
    else:
        return 0.0

    raw_size = TARGET_VOL / (recent_vol + 1e-8)
    return round(direction * min(raw_size, MAX_POSITION), 4)


def predict(ticker: str) -> dict:
    model, scaler, n_features = load_model(ticker)
    context_tensor, df, features = prepare_context(ticker, scaler, n_features)

    with torch.no_grad():
        logit        = model(context_tensor)
        sigmoid_out  = float(torch.sigmoid(logit).item())

    last_close = float(df["Close"].iloc[-1])
    last_date  = df.index[-1]

    log_ret    = np.log(df["Close"] / df["Close"].shift(1)).dropna()
    recent_vol = float(log_ret.iloc[-20:].std())

    position   = compute_position(sigmoid_out, recent_vol)
    stance     = "LONG" if position > 0 else "SHORT" if position < 0 else "FLAT"

    # 5-day price range — symmetric random-walk bands, no directional content
    forecast_dates = []
    current = last_date
    while len(forecast_dates) < 5:
        current += timedelta(days=1)
        if current.weekday() < 5:
            forecast_dates.append(current)

    forecast = []
    for i, d in enumerate(forecast_dates):
        hv = recent_vol * np.sqrt(i + 1)
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
        "sigmoid_output":   round(sigmoid_out, 4),
        "stance":           stance,
        "position_size":    abs(position),
        "daily_volatility": round(recent_vol * 100, 3),
        "forecast":         forecast,
        # Legacy key — kept for API/frontend compatibility
        "regime_prob":      round(sigmoid_out, 4),
    }


def print_forecast(result: dict):
    print(f"\n{'='*60}")
    print(f"  StckMind — {result['ticker']}")
    print(f"{'='*60}")
    print(f"  Last close     : ${result['last_known_close']}  ({result['last_known_date']})")
    print(f"  Sigmoid output : {result['sigmoid_output']}  (raw, uncalibrated)")
    print(f"  Stance         : {result['stance']}")
    print(f"  Position size  : {result['position_size']:.2%}")
    print(f"  Daily vol      : {result['daily_volatility']}%")
    print(f"\n  5-Day 95% Range (symmetric — no directional content):")
    print(f"  {'Date':<12}  {'Low':>9}  {'High':>9}")
    print(f"  {'-'*36}")
    for row in result["forecast"]:
        print(f"  {row['date']:<12}  ${row['low_95']:>8}  ${row['high_95']:>8}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    result = predict("AAPL")
    print_forecast(result)