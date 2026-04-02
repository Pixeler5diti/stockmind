import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator
from ta.trend import MACD, EMAIndicator, SMAIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice
import os
import warnings
warnings.filterwarnings("ignore")

START_DATE  = "2022-01-01"
DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "feature_store")
os.makedirs(DATA_DIR, exist_ok=True)

FEATURES = [
    "Open", "High", "Low", "Close", "Volume",
    "EMA9", "EMA21", "EMA50",
    "SMA20", "SMA50",
    "MACD", "MACD_signal", "MACD_diff",
    "ADX",
    "RSI14", "RSI7",
    "Stoch_k", "Stoch_d",
    "ROC10",
    "BB_upper", "BB_lower", "BB_width", "BB_pct",
    "ATR14",
    "OBV",
    "VWAP",
    "High_Low_pct",
    "Close_Open_pct",
    "Price_vs_SMA20",
    "Price_vs_SMA50",
]

# Target columns produced by add_targets()
# These are NOT model inputs — they are what the model learns to predict
TARGETS = ["realized_vol_5d", "trend_regime"]


def fetch_data(ticker: str, start: str = START_DATE) -> pd.DataFrame:
    print(f"[+] Fetching data for {ticker} from {start}...")
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data found for ticker: {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)
    print(f"    {len(df)} rows fetched ({df.index[0].date()} to {df.index[-1].date()})")
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    print("[+] Engineering features...")
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    vol   = df["Volume"]

    df["EMA9"]        = EMAIndicator(close, window=9).ema_indicator()
    df["EMA21"]       = EMAIndicator(close, window=21).ema_indicator()
    df["EMA50"]       = EMAIndicator(close, window=50).ema_indicator()
    df["SMA20"]       = SMAIndicator(close, window=20).sma_indicator()
    df["SMA50"]       = SMAIndicator(close, window=50).sma_indicator()

    macd              = MACD(close)
    df["MACD"]        = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["MACD_diff"]   = macd.macd_diff()
    df["ADX"]         = ADXIndicator(high, low, close).adx()

    df["RSI14"]       = RSIIndicator(close, window=14).rsi()
    df["RSI7"]        = RSIIndicator(close, window=7).rsi()

    stoch             = StochasticOscillator(high, low, close)
    df["Stoch_k"]     = stoch.stoch()
    df["Stoch_d"]     = stoch.stoch_signal()
    df["ROC10"]       = ROCIndicator(close, window=10).roc()

    bb                = BollingerBands(close, window=20)
    df["BB_upper"]    = bb.bollinger_hband()
    df["BB_lower"]    = bb.bollinger_lband()
    df["BB_width"]    = bb.bollinger_wband()
    df["BB_pct"]      = bb.bollinger_pband()
    df["ATR14"]       = AverageTrueRange(high, low, close, window=14).average_true_range()

    df["OBV"]         = OnBalanceVolumeIndicator(close, vol).on_balance_volume()
    try:
        df["VWAP"]    = VolumeWeightedAveragePrice(high, low, close, vol).volume_weighted_average_price()
    except Exception:
        df["VWAP"]    = close.rolling(14).mean()

    df["High_Low_pct"]   = (high - low) / close * 100
    df["Close_Open_pct"] = (close - df["Open"]) / df["Open"] * 100
    df["Price_vs_SMA20"] = (close - df["SMA20"]) / df["SMA20"] * 100
    df["Price_vs_SMA50"] = (close - df["SMA50"]) / df["SMA50"] * 100

    df.dropna(inplace=True)
    print(f"    {len(FEATURES)} features ready — {len(df)} clean rows")
    return df


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add prediction targets to the dataframe.

    Target 1 — realized_vol_5d (regression target)
        Standard deviation of the NEXT 5 daily log returns.
        Tells the model how turbulent the next 5 days will be.
        Volatility clusters — high vol today predicts high vol tomorrow.
        This is learnable with daily technical data.

    Target 2 — trend_regime (classification target)
        Binary label: 1 if price is above its 20-day SMA, 0 if below.
        Simple proxy for whether the stock is in an uptrend or downtrend.
        Used by the strategy to decide direction (long vs short).

    Target 3 — forward_return_1d (NOT a training target)
        Actual next-day log return.
        Stored in the dataframe for backtest simulation only.
        Never passed to the model as a label.
    """
    print("[+] Computing targets...")

    log_ret = np.log(df["Close"] / df["Close"].shift(1))

    # realized_vol_5d: rolling std of next 5 returns
    # shift(-1) moves forward one day (exclude today)
    # rolling(5) takes 5-day window
    # shift(-4) aligns the window so it covers days t+1 through t+5
    df["realized_vol_5d"] = (
        log_ret.shift(-1)
               .rolling(5)
               .std()
               .shift(-4)
    )

    # trend_regime: 1 = price above SMA20 (uptrend), 0 = below (downtrend)
    df["trend_regime"] = (df["Close"] > df["SMA20"]).astype(int)

    # forward_return_1d: actual next-day log return (backtest use only)
    df["forward_return_1d"] = log_ret.shift(-1)

    # Drop rows where targets are NaN (last 5 rows will be NaN due to shifting)
    before = len(df)
    df.dropna(subset=["realized_vol_5d", "forward_return_1d"], inplace=True)
    print(f"    Targets added — {before - len(df)} rows dropped (lookahead window)")

    # Print target distribution for sanity check
    print(f"    realized_vol_5d  — mean: {df['realized_vol_5d'].mean():.5f}  std: {df['realized_vol_5d'].std():.5f}")
    print(f"    trend_regime     — uptrend: {df['trend_regime'].mean()*100:.1f}%  downtrend: {(1-df['trend_regime'].mean())*100:.1f}%")

    return df


def save_features(df: pd.DataFrame, ticker: str) -> str:
    path = os.path.join(DATA_DIR, f"{ticker.lower()}_features.parquet")
    df.to_parquet(path)
    print(f"[+] Saved → {path}")
    return path


def run_pipeline(ticker: str) -> pd.DataFrame:
    df = fetch_data(ticker)
    df = add_indicators(df)
    df = add_targets(df)      # new — compute vol + regime targets
    save_features(df, ticker)
    print(f"[✓] Pipeline done for {ticker}\n")
    return df


if __name__ == "__main__":
    for ticker in ["^GSPC", "AAPL"]:
        df = run_pipeline(ticker)
        print(df[FEATURES + TARGETS].tail(5))
        print()