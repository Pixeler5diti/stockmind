"""
backtest.py — Walk-forward backtest using regime signal + volatility-targeted sizing.

Strategy:
  - Each day: use last 30 days of features to get regime probability
  - If P(uptrend) > 0.65: go long, sized by vol target
  - If P(uptrend) < 0.35: go short, sized by vol target
  - Else: stay flat
  - Position size = TARGET_VOL / realized_vol_20d (capped at 1x)
  - Transaction cost applied on position change magnitude

Metrics:
  - Sharpe, Sortino, Max Drawdown, Win Rate, Profit Factor
  - vs Buy-and-Hold baseline
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
import pandas as pd
import joblib
import json

from models.lstm_model import StockLSTM, FEATURES, CONTEXT_LEN
from models.predict import compute_position

OUTPUTS_DIR   = os.path.join(os.path.dirname(__file__), "..", "outputs")
BACKTEST_DIR  = os.path.join(os.path.dirname(__file__), "..", "outputs", "backtest")
FEATURE_STORE = os.path.join(os.path.dirname(__file__), "..", "feature_store")
os.makedirs(BACKTEST_DIR, exist_ok=True)

INITIAL_CAP   = 10_000.0
TRANSACTION   = 0.001       # 0.1% per unit of position change
TRADING_DAYS  = 252
VOL_WINDOW    = 20          # days for realized vol estimate


def load_model(ticker: str):
    out_dir          = os.path.join(OUTPUTS_DIR, ticker.lower())
    model_path       = os.path.join(out_dir, f"{ticker.lower()}_child_model.pt")
    feat_scaler_path = os.path.join(out_dir, f"{ticker.lower()}_feat_scaler.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model for {ticker}. Run train.py first.")

    model = StockLSTM(input_size=len(FEATURES))
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    feat_scaler = joblib.load(feat_scaler_path)
    return model, feat_scaler


def load_data(ticker: str) -> pd.DataFrame:
    path = os.path.join(FEATURE_STORE, f"{ticker.lower()}_features.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No feature data for {ticker}. Run data_pipeline.py first.")
    df = pd.read_parquet(path)
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df.dropna(inplace=True)
    return df


# ── Quant Metrics ──────────────────────────────────────

def sharpe(returns, periods=TRADING_DAYS):
    if returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(periods))


def sortino(returns, periods=TRADING_DAYS):
    downside = returns[returns < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    return float(returns.mean() / downside.std() * np.sqrt(periods))


def max_drawdown(equity):
    peak = np.maximum.accumulate(equity)
    dd   = (equity - peak) / peak
    return float(dd.min())


def annualized_return(total_ret, n_days):
    return float((1 + total_ret) ** (TRADING_DAYS / n_days) - 1)


def profit_factor(returns):
    gains  = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    return float(gains / losses) if losses > 0 else float("inf")


# ── Backtest Engine ────────────────────────────────────

def run_backtest(ticker: str) -> dict:
    print(f"\n{'='*60}")
    print(f"  BACKTEST — {ticker}")
    print(f"{'='*60}")

    model, feat_scaler = load_model(ticker)
    df                 = load_data(ticker)

    # Walk-forward: test on last 20% of data
    split_idx = int(len(df) * 0.8)
    n_test    = len(df) - split_idx

    print(f"  Test window : {df.index[split_idx].date()} → {df.index[-1].date()}")
    print(f"  Test days   : {n_test}")

    portfolio     = INITIAL_CAP
    bnh_portfolio = INITIAL_CAP
    prev_position = 0.0

    strat_returns = []
    bnh_returns   = []
    equity        = [INITIAL_CAP]
    bnh_equity    = [INITIAL_CAP]
    positions     = []
    regime_probs  = []
    dates         = []

    for i in range(split_idx, len(df) - 1):
        if i < CONTEXT_LEN + VOL_WINDOW:
            continue

        # Feature context window
        window     = df.iloc[i - CONTEXT_LEN:i][FEATURES].values
        window_sc  = feat_scaler.transform(window)
        tensor     = torch.tensor(window_sc, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            _, pred_regime = model(tensor)

        regime_prob = float(pred_regime.item())

        # Realized vol from last VOL_WINDOW days (more stable than model vol)
        recent_vol  = float(df["log_return"].iloc[i - VOL_WINDOW:i].std())

        # Signal
        position    = compute_position(regime_prob, recent_vol)

        # Actual next-day return
        actual_log  = float(df["log_return"].iloc[i + 1])
        actual_pct  = np.exp(actual_log) - 1

        # Strategy return with transaction cost on position change
        pos_change  = abs(position - prev_position)
        strat_ret   = position * actual_pct - pos_change * TRANSACTION

        prev_position = position

        portfolio     *= (1 + strat_ret)
        bnh_portfolio *= (1 + actual_pct)

        strat_returns.append(strat_ret)
        bnh_returns.append(actual_pct)
        equity.append(portfolio)
        bnh_equity.append(bnh_portfolio)
        positions.append(position)
        regime_probs.append(regime_prob)
        dates.append(df.index[i])

    # ── Compute metrics ────────────────────────────────
    strat_arr  = np.array(strat_returns)
    bnh_arr    = np.array(bnh_returns)
    eq_arr     = np.array(equity)
    bnh_eq_arr = np.array(bnh_equity)
    pos_arr    = np.array(positions)
    n_days     = len(strat_arr)

    total_ret  = (portfolio - INITIAL_CAP) / INITIAL_CAP
    bnh_ret    = (bnh_portfolio - INITIAL_CAP) / INITIAL_CAP

    wins       = np.sum(strat_arr > 0)
    losses     = np.sum(strat_arr < 0)
    win_rate   = wins / (wins + losses) if (wins + losses) > 0 else 0

    n_long     = int(np.sum(pos_arr > 0))
    n_short    = int(np.sum(pos_arr < 0))
    n_flat     = int(np.sum(pos_arr == 0))

    avg_regime = float(np.mean(regime_probs))

    metrics = {
        "ticker":                ticker,
        "test_start":            str(dates[0].date()) if dates else "N/A",
        "test_end":              str(dates[-1].date()) if dates else "N/A",
        "n_days":                n_days,
        "initial_capital":       INITIAL_CAP,
        "final_capital":         round(portfolio, 2),
        "total_return_pct":      round(total_ret * 100, 3),
        "annualized_return_pct": round(annualized_return(total_ret, n_days) * 100, 3),
        "sharpe_ratio":          round(sharpe(strat_arr), 4),
        "sortino_ratio":         round(sortino(strat_arr), 4),
        "max_drawdown_pct":      round(max_drawdown(eq_arr) * 100, 3),
        "win_rate_pct":          round(win_rate * 100, 2),
        "profit_factor":         round(profit_factor(strat_arr), 4),
        "n_long":                n_long,
        "n_short":               n_short,
        "n_flat":                n_flat,
        "avg_regime_prob":       round(avg_regime, 4),
        "bnh_total_return_pct":  round(bnh_ret * 100, 3),
        "bnh_sharpe":            round(sharpe(bnh_arr), 4),
        "bnh_max_drawdown_pct":  round(max_drawdown(bnh_eq_arr) * 100, 3),
        "alpha_pct":             round((total_ret - bnh_ret) * 100, 3),
        "equity_curve":          [round(v, 2) for v in equity],
        "bnh_curve":             [round(v, 2) for v in bnh_equity],
        "dates":                 [str(d.date()) for d in dates],
    }

    _print_metrics(metrics)
    _save_metrics(metrics, ticker)
    return metrics


def _print_metrics(m):
    print(f"\n  --- Strategy ---")
    print(f"  Total return      : {m['total_return_pct']:>+8.3f}%")
    print(f"  Annualized return : {m['annualized_return_pct']:>+8.3f}%")
    print(f"  Sharpe ratio      : {m['sharpe_ratio']:>8.4f}")
    print(f"  Sortino ratio     : {m['sortino_ratio']:>8.4f}")
    print(f"  Max drawdown      : {m['max_drawdown_pct']:>8.3f}%")
    print(f"  Win rate          : {m['win_rate_pct']:>8.2f}%")
    print(f"  Profit factor     : {m['profit_factor']:>8.4f}")
    print(f"  Positions (L/S/F) : {m['n_long']} / {m['n_short']} / {m['n_flat']}")
    print(f"  Avg regime prob   : {m['avg_regime_prob']:>8.4f}")
    print(f"\n  --- Buy & Hold ---")
    print(f"  Total return      : {m['bnh_total_return_pct']:>+8.3f}%")
    print(f"  Sharpe ratio      : {m['bnh_sharpe']:>8.4f}")
    print(f"  Max drawdown      : {m['bnh_max_drawdown_pct']:>8.3f}%")
    print(f"\n  --- Alpha ---")
    print(f"  Strategy vs BnH   : {m['alpha_pct']:>+8.3f}%")
    print(f"  Final capital     : ${m['final_capital']:,.2f}")
    print(f"{'='*60}\n")


def _save_metrics(m, ticker):
    summary = {k: v for k, v in m.items()
               if k not in ("equity_curve", "bnh_curve", "dates")}
    path = os.path.join(BACKTEST_DIR, f"{ticker.lower()}_backtest.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  [saved] {path}")


if __name__ == "__main__":
    run_backtest("AAPL")