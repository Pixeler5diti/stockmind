"""
backtest.py — Walk-forward backtesting for StckMind LSTM predictions

Strategy:
  - For each day in the test window, use the previous 60 days to predict next day return
  - Go LONG if predicted return > threshold, SHORT if < -threshold, else FLAT
  - Track portfolio value, compare vs buy-and-hold baseline

Metrics reported:
  - Total return
  - Annualized return
  - Sharpe ratio (annualized, risk-free = 0)
  - Sortino ratio
  - Max drawdown
  - Win rate
  - Profit factor
  - Buy-and-hold comparison
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
import pandas as pd
import joblib
import json
from datetime import datetime

from models.lstm_model import StockLSTM, FEATURES, CONTEXT_LEN
from data.data_pipeline import add_indicators, FEATURES as FEAT_LIST

OUTPUTS_DIR   = os.path.join(os.path.dirname(__file__), "..", "outputs")
BACKTEST_DIR  = os.path.join(os.path.dirname(__file__), "..", "outputs", "backtest")
os.makedirs(BACKTEST_DIR, exist_ok=True)

TRADING_DAYS  = 252
INITIAL_CAP   = 10_000.0      # $10,000 starting capital
THRESHOLD     = 0.0003        # minimum predicted return to take a position (~0.03%)
TRANSACTION   = 0.001         # 0.1% transaction cost per trade


def load_model_and_scaler(ticker: str):
    out_dir    = os.path.join(OUTPUTS_DIR, ticker.lower())
    model_path = os.path.join(out_dir, f"{ticker.lower()}_child_model.pt")
    scaler_path= os.path.join(out_dir, f"{ticker.lower()}_scaler.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found for {ticker}. Run train.py first.")

    model = StockLSTM(input_size=len(FEATURES))
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    scaler = joblib.load(scaler_path)
    return model, scaler


def load_test_data(ticker: str) -> pd.DataFrame:
    """Load parquet and return the most recent 20% as test window (walk-forward)."""
    parquet_path = os.path.join(
        os.path.dirname(__file__), "..", "feature_store",
        f"{ticker.lower()}_features.parquet"
    )
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"No feature data for {ticker}. Run data_pipeline.py first.")

    df = pd.read_parquet(parquet_path)

    # Add log returns
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df.dropna(inplace=True)

    # Test window = last 20%
    split = int(len(df) * 0.8)
    return df.iloc[split:].copy()


# ── Core Metrics ───────────────────────────────────────

def sharpe_ratio(returns: np.ndarray, periods: int = TRADING_DAYS) -> float:
    """Annualized Sharpe ratio assuming risk-free rate = 0."""
    if returns.std() == 0:
        return 0.0
    return float((returns.mean() / returns.std()) * np.sqrt(periods))


def sortino_ratio(returns: np.ndarray, periods: int = TRADING_DAYS) -> float:
    """Annualized Sortino ratio — penalizes downside volatility only."""
    downside = returns[returns < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    return float((returns.mean() / downside.std()) * np.sqrt(periods))


def max_drawdown(equity_curve: np.ndarray) -> float:
    """Maximum peak-to-trough decline as a fraction."""
    peak    = np.maximum.accumulate(equity_curve)
    dd      = (equity_curve - peak) / peak
    return float(dd.min())


def annualized_return(total_return: float, n_days: int) -> float:
    return float((1 + total_return) ** (TRADING_DAYS / n_days) - 1)


def profit_factor(returns: np.ndarray) -> float:
    gains  = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    return float(gains / losses) if losses > 0 else float("inf")


# ── Walk-Forward Backtest ──────────────────────────────

def run_backtest(ticker: str, threshold: float = THRESHOLD) -> dict:
    print(f"\n{'='*60}")
    print(f"  BACKTEST — {ticker}")
    print(f"{'='*60}")

    model, scaler  = load_model_and_scaler(ticker)
    test_df        = load_test_data(ticker)

    # We need CONTEXT_LEN days before the test window for the first prediction
    full_parquet = os.path.join(
        os.path.dirname(__file__), "..", "feature_store",
        f"{ticker.lower()}_features.parquet"
    )
    full_df       = pd.read_parquet(full_parquet)
    full_df["log_return"] = np.log(full_df["Close"] / full_df["Close"].shift(1))
    full_df.dropna(inplace=True)

    split_idx     = int(len(full_df) * 0.8)
    test_start    = split_idx
    n_test        = len(full_df) - test_start

    print(f"  Test window : {full_df.index[test_start].date()} → {full_df.index[-1].date()}")
    print(f"  Test days   : {n_test}")

    # ── Simulate day by day ─────────────────────────────
    portfolio     = INITIAL_CAP
    bnh_portfolio = INITIAL_CAP     # buy and hold baseline

    strategy_returns = []
    bnh_returns      = []
    equity_curve     = [INITIAL_CAP]
    bnh_curve        = [INITIAL_CAP]
    signals          = []           # +1 long, -1 short, 0 flat
    dates            = []

    prev_signal      = 0            # track position changes for transaction cost

    for i in range(n_test):
        idx       = test_start + i
        if idx < CONTEXT_LEN:
            continue

        # Context window: last 60 rows before this day
        window    = full_df.iloc[idx - CONTEXT_LEN : idx][FEATURES].values
        window_sc = scaler.transform(window)
        tensor    = torch.tensor(window_sc, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            pred_returns = model(tensor).numpy().flatten()

        # Use only day-1 prediction for signal
        pred_day1 = float(np.clip(pred_returns[0], -0.05, 0.05))

        # Signal generation
        if pred_day1 > threshold:
            signal = 1       # long
        elif pred_day1 < -threshold:
            signal = -1      # short
        else:
            signal = 0       # flat

        # Actual return that day
        actual_return = float(full_df["log_return"].iloc[idx])
        actual_pct    = np.exp(actual_return) - 1

        # Strategy return
        strat_return  = signal * actual_pct

        # Transaction cost on position change
        if signal != prev_signal and signal != 0:
            strat_return -= TRANSACTION

        prev_signal   = signal

        portfolio     *= (1 + strat_return)
        bnh_portfolio *= (1 + actual_pct)

        strategy_returns.append(strat_return)
        bnh_returns.append(actual_pct)
        equity_curve.append(portfolio)
        bnh_curve.append(bnh_portfolio)
        signals.append(signal)
        dates.append(full_df.index[idx])

    # ── Compute metrics ─────────────────────────────────
    strat_arr   = np.array(strategy_returns)
    bnh_arr     = np.array(bnh_returns)
    eq_arr      = np.array(equity_curve)
    bnh_eq_arr  = np.array(bnh_curve)
    n_days      = len(strat_arr)

    total_ret   = (portfolio - INITIAL_CAP) / INITIAL_CAP
    bnh_ret     = (bnh_portfolio - INITIAL_CAP) / INITIAL_CAP

    wins        = np.sum(strat_arr > 0)
    losses      = np.sum(strat_arr < 0)
    win_rate    = wins / (wins + losses) if (wins + losses) > 0 else 0

    longs       = np.sum(np.array(signals) == 1)
    shorts      = np.sum(np.array(signals) == -1)
    flats       = np.sum(np.array(signals) == 0)

    metrics = {
        "ticker":              ticker,
        "test_start":          str(dates[0].date()) if dates else "N/A",
        "test_end":            str(dates[-1].date()) if dates else "N/A",
        "n_days":              n_days,
        "initial_capital":     INITIAL_CAP,
        "final_capital":       round(portfolio, 2),
        "total_return_pct":    round(total_ret * 100, 3),
        "annualized_return_pct": round(annualized_return(total_ret, n_days) * 100, 3),
        "sharpe_ratio":        round(sharpe_ratio(strat_arr), 4),
        "sortino_ratio":       round(sortino_ratio(strat_arr), 4),
        "max_drawdown_pct":    round(max_drawdown(eq_arr) * 100, 3),
        "win_rate_pct":        round(win_rate * 100, 2),
        "profit_factor":       round(profit_factor(strat_arr), 4),
        "n_long":              int(longs),
        "n_short":             int(shorts),
        "n_flat":              int(flats),
        "bnh_total_return_pct": round(bnh_ret * 100, 3),
        "bnh_sharpe":          round(sharpe_ratio(bnh_arr), 4),
        "bnh_max_drawdown_pct": round(max_drawdown(bnh_eq_arr) * 100, 3),
        "alpha_pct":           round((total_ret - bnh_ret) * 100, 3),
        "equity_curve":        [round(v, 2) for v in equity_curve],
        "bnh_curve":           [round(v, 2) for v in bnh_curve],
        "dates":               [str(d.date()) for d in dates],
    }

    _print_metrics(metrics)
    _save_metrics(metrics, ticker)

    return metrics


def _print_metrics(m: dict):
    print(f"\n  --- Strategy ---")
    print(f"  Total return      : {m['total_return_pct']:>8.3f}%")
    print(f"  Annualized return : {m['annualized_return_pct']:>8.3f}%")
    print(f"  Sharpe ratio      : {m['sharpe_ratio']:>8.4f}")
    print(f"  Sortino ratio     : {m['sortino_ratio']:>8.4f}")
    print(f"  Max drawdown      : {m['max_drawdown_pct']:>8.3f}%")
    print(f"  Win rate          : {m['win_rate_pct']:>8.2f}%")
    print(f"  Profit factor     : {m['profit_factor']:>8.4f}")
    print(f"  Positions (L/S/F) : {m['n_long']} / {m['n_short']} / {m['n_flat']}")
    print(f"\n  --- Buy & Hold Baseline ---")
    print(f"  Total return      : {m['bnh_total_return_pct']:>8.3f}%")
    print(f"  Sharpe ratio      : {m['bnh_sharpe']:>8.4f}")
    print(f"  Max drawdown      : {m['bnh_max_drawdown_pct']:>8.3f}%")
    print(f"\n  --- Alpha ---")
    print(f"  Strategy vs BnH   : {m['alpha_pct']:>+8.3f}%")
    print(f"  Final capital     : ${m['final_capital']:,.2f}  (started ${m['initial_capital']:,.2f})")
    print(f"{'='*60}\n")


def _save_metrics(m: dict, ticker: str):
    path = os.path.join(BACKTEST_DIR, f"{ticker.lower()}_backtest.json")
    # Save without equity curves for clean JSON summary
    summary = {k: v for k, v in m.items() if k not in ("equity_curve", "bnh_curve", "dates")}
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  [saved] {path}")


if __name__ == "__main__":
    run_backtest("AAPL")