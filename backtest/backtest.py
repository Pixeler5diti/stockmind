"""
backtest.py — Vol-scaled long-only strategy.

Core idea:
  Always long the asset. Scale position based on volatility direction signal.
  When vol is rising (risky) → reduce exposure.
  When vol is falling (calm) → increase exposure.
  This is risk management, not direction prediction.

Position logic:
  base_position = 1.0
  vol_prob = sigmoid output of vol_direction model

  raw_position = 1.0 - 0.5 * (vol_prob - 0.5) * 2
    → vol_prob=1.0 (very high vol) → position=0.5
    → vol_prob=0.5 (uncertain)     → position=1.0
    → vol_prob=0.0 (very low vol)  → position=1.5

  Confidence filter:
    Only update position if vol_prob > 0.6 or vol_prob < 0.4
    Otherwise hold current position (avoid noise)

  EMA smoothing (span=5) applied to raw positions
  Clipped to [0.0, 1.5]
  Transaction cost: 0.02% (0.0002) per unit of position change

No lookahead:
  Signal at close of day t → applied to return of day t+1
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
import pandas as pd
import joblib
import json

from models.lstm_model import StockLSTM, CONTEXT_LEN

OUTPUTS_DIR   = os.path.join(os.path.dirname(__file__), "..", "outputs")
BACKTEST_DIR  = os.path.join(os.path.dirname(__file__), "..", "outputs", "backtest")
FEATURE_STORE = os.path.join(os.path.dirname(__file__), "..", "feature_store")
os.makedirs(BACKTEST_DIR, exist_ok=True)

INITIAL_CAP  = 10_000.0
TRANSACTION  = 0.0002      # 0.02% per unit of position change
TRADING_DAYS = 252
VOL_WINDOW   = 20
CONF_HIGH    = 0.6         # only act if vol_prob above this...
CONF_LOW     = 0.4         # ...or below this
POS_MIN      = 0.0
POS_MAX      = 1.5
EMA_SPAN     = 5


def load_vol_model(ticker):
    out_dir     = os.path.join(OUTPUTS_DIR, ticker.lower())
    model_path  = os.path.join(out_dir, "vol_model.pt")
    scaler_path = os.path.join(out_dir, "vol_model_scaler.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No vol_model for {ticker}. Run train.py first.")

    scaler     = joblib.load(scaler_path)
    n_features = scaler.n_features_in_
    model      = StockLSTM(input_size=n_features)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model, scaler


def get_features(df):
    exclude = {"vol_regime", "vol_direction", "price_direction",
               "forward_return_1d", "Open", "High", "Low", "Close", "Volume"}
    return [c for c in df.columns if c not in exclude]


def load_data(ticker):
    path = os.path.join(FEATURE_STORE, f"{ticker.lower()}_features.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No feature data for {ticker}.")
    df = pd.read_parquet(path)
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df.dropna(inplace=True)
    return df


def ema_smooth(series, span=EMA_SPAN):
    """Exponential moving average for position smoothing."""
    alpha  = 2 / (span + 1)
    result = [series[0]]
    for v in series[1:]:
        result.append(alpha * v + (1 - alpha) * result[-1])
    return np.array(result)


# ── Metrics ────────────────────────────────────────────

def sharpe(returns, periods=TRADING_DAYS):
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(periods))


def sortino(returns, periods=TRADING_DAYS):
    down = returns[returns < 0]
    if len(down) == 0 or down.std() == 0:
        return 0.0
    return float(returns.mean() / down.std() * np.sqrt(periods))


def max_drawdown(equity):
    peak = np.maximum.accumulate(equity)
    dd   = (equity - peak) / peak
    return float(dd.min())


def annualized_return(total_ret, n_days):
    return float((1 + total_ret) ** (TRADING_DAYS / n_days) - 1)


# ── Backtest engine ────────────────────────────────────

def run_backtest(ticker: str) -> dict:
    print(f"\n{'='*60}")
    print(f"  VOL-SCALED LONG BACKTEST — {ticker}")
    print(f"{'='*60}")

    vol_model, vol_scaler = load_vol_model(ticker)
    df       = load_data(ticker)
    features = get_features(df)

    df_feat          = df.copy()
    df_feat[features] = df_feat[features].replace([np.inf, -np.inf], np.nan)
    df_feat.dropna(inplace=True)

    split_idx = int(len(df_feat) * 0.8)
    n_test    = len(df_feat) - split_idx

    print(f"  Test window : {df_feat.index[split_idx].date()} → {df_feat.index[-1].date()}")
    print(f"  Test days   : {n_test}")

    n_features = vol_scaler.n_features_in_

    # ── Pass 1: generate raw positions ─────────────────
    raw_positions = []
    vol_probs_all = []
    current_pos   = 1.0   # start fully long

    for i in range(split_idx, len(df_feat) - 1):
        if i < CONTEXT_LEN + VOL_WINDOW:
            raw_positions.append(1.0)
            vol_probs_all.append(0.5)
            continue

        window     = df_feat.iloc[i - CONTEXT_LEN:i][features].values
        window_sc  = vol_scaler.transform(window[:, :n_features])
        tensor     = torch.tensor(window_sc, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            vol_prob = float(torch.sigmoid(vol_model(tensor)).item())

        # Confidence filter — only update if model is decisive
        if vol_prob > CONF_HIGH or vol_prob < CONF_LOW:
            # Continuous scaling: high vol_prob → reduce, low vol_prob → increase
            raw_pos = 1.0 - 0.5 * (vol_prob - 0.5) * 2
            raw_pos = float(np.clip(raw_pos, POS_MIN, POS_MAX))
            current_pos = raw_pos
        # else: hold current_pos (no change when uncertain)

        raw_positions.append(current_pos)
        vol_probs_all.append(vol_prob)

    # ── Pass 2: EMA smooth positions ───────────────────
    raw_arr      = np.array(raw_positions)
    smooth_pos   = ema_smooth(raw_arr, span=EMA_SPAN)
    smooth_pos   = np.clip(smooth_pos, POS_MIN, POS_MAX)

    # ── Pass 3: simulate returns ────────────────────────
    portfolio     = INITIAL_CAP
    bnh_portfolio = INITIAL_CAP
    prev_position = 1.0

    strat_returns = []
    bnh_returns   = []
    equity        = [INITIAL_CAP]
    bnh_eq        = [INITIAL_CAP]
    dates         = []
    turnover_list = []

    for j, i in enumerate(range(split_idx, len(df_feat) - 1)):
        if j >= len(smooth_pos):
            break

        position   = smooth_pos[j]
        actual_log = float(df_feat["log_return"].iloc[i + 1])
        actual_pct = np.exp(actual_log) - 1

        pos_change = abs(position - prev_position)
        strat_ret  = position * actual_pct - pos_change * TRANSACTION
        prev_position = position

        portfolio     *= (1 + strat_ret)
        bnh_portfolio *= (1 + actual_pct)

        strat_returns.append(strat_ret)
        bnh_returns.append(actual_pct)
        equity.append(portfolio)
        bnh_eq.append(bnh_portfolio)
        turnover_list.append(pos_change)
        dates.append(df_feat.index[i])

    strat_arr = np.array(strat_returns)
    bnh_arr   = np.array(bnh_returns)
    n_days    = len(strat_arr)

    if n_days == 0:
        return {"error": "Not enough test data"}

    total_ret  = (portfolio - INITIAL_CAP) / INITIAL_CAP
    bnh_ret    = (bnh_portfolio - INITIAL_CAP) / INITIAL_CAP
    avg_turnover = float(np.mean(turnover_list))

    wins     = np.sum(strat_arr > 0)
    losses   = np.sum(strat_arr < 0)
    win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0

    pf_gains  = strat_arr[strat_arr > 0].sum()
    pf_losses = abs(strat_arr[strat_arr < 0].sum())
    profit_factor = float(pf_gains / pf_losses) if pf_losses > 0 else float("inf")

    avg_pos = float(np.mean(smooth_pos))
    pos_above1 = float(np.mean(smooth_pos > 1.0) * 100)
    pos_below1 = float(np.mean(smooth_pos < 1.0) * 100)

    metrics = {
        "ticker":                ticker,
        "strategy":              "vol_scaled_long",
        "test_start":            str(dates[0].date()) if dates else "N/A",
        "test_end":              str(dates[-1].date()) if dates else "N/A",
        "n_days":                n_days,
        "initial_capital":       INITIAL_CAP,
        "final_capital":         round(portfolio, 2),
        "total_return_pct":      round(total_ret * 100, 3),
        "annualized_return_pct": round(annualized_return(total_ret, n_days) * 100, 3),
        "sharpe_ratio":          round(sharpe(strat_arr), 4),
        "sortino_ratio":         round(sortino(strat_arr), 4),
        "max_drawdown_pct":      round(max_drawdown(np.array(equity)) * 100, 3),
        "win_rate_pct":          round(win_rate * 100, 2),
        "profit_factor":         round(profit_factor, 4),
        "avg_position":          round(avg_pos, 3),
        "pct_overweight":        round(pos_above1, 1),
        "pct_underweight":       round(pos_below1, 1),
        "avg_daily_turnover":    round(avg_turnover, 5),
        "bnh_total_return_pct":  round(bnh_ret * 100, 3),
        "bnh_sharpe":            round(sharpe(bnh_arr), 4),
        "bnh_max_drawdown_pct":  round(max_drawdown(np.array(bnh_eq)) * 100, 3),
        "alpha_pct":             round((total_ret - bnh_ret) * 100, 3),
        "equity_curve":          [round(v, 2) for v in equity],
        "bnh_curve":             [round(v, 2) for v in bnh_eq],
        "positions":             [round(v, 4) for v in smooth_pos[:len(dates)]],
        "vol_series":            [round(float(df_feat["log_return"].iloc[split_idx+j-VOL_WINDOW:split_idx+j].std()), 6) for j in range(len(dates))],
        "dates":                 [str(d.date()) for d in dates],
    }

    _print_metrics(metrics)
    _save_metrics(metrics, ticker)
    return metrics


def _print_metrics(m):
    print(f"\n  --- Strategy: Vol-Scaled Long ---")
    print(f"  Total return      : {m['total_return_pct']:>+8.3f}%")
    print(f"  Annualized return : {m['annualized_return_pct']:>+8.3f}%")
    print(f"  Sharpe ratio      : {m['sharpe_ratio']:>8.4f}")
    print(f"  Sortino ratio     : {m['sortino_ratio']:>8.4f}")
    print(f"  Max drawdown      : {m['max_drawdown_pct']:>8.3f}%")
    print(f"  Win rate          : {m['win_rate_pct']:>8.2f}%")
    print(f"  Profit factor     : {m['profit_factor']:>8.4f}")
    print(f"  Avg position      : {m['avg_position']:>8.3f}x")
    print(f"  Overweight days   : {m['pct_overweight']:>7.1f}%")
    print(f"  Underweight days  : {m['pct_underweight']:>7.1f}%")
    print(f"  Avg daily turnover: {m['avg_daily_turnover']:>8.5f}")
    print(f"\n  --- Buy & Hold (1.0x always) ---")
    print(f"  Total return      : {m['bnh_total_return_pct']:>+8.3f}%")
    print(f"  Sharpe ratio      : {m['bnh_sharpe']:>8.4f}")
    print(f"  Max drawdown      : {m['bnh_max_drawdown_pct']:>8.3f}%")
    print(f"\n  --- Comparison ---")
    print(f"  Alpha vs BnH      : {m['alpha_pct']:>+8.3f}%")
    print(f"  Final capital     : ${m['final_capital']:,.2f}")
    beats = "BEATS BUY & HOLD ✓" if m['sharpe_ratio'] > m['bnh_sharpe'] else "UNDERPERFORMS BUY & HOLD"
    print(f"  Sharpe comparison : {beats}")
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