import yfinance as yf
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

START_DATE = "2020-01-01"
DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "feature_store")
os.makedirs(DATA_DIR, exist_ok=True)

FEATURES = [
    "log_return",
    "abs_return",
    "vol_5",
    "vol_10",
    "vol_20",
    "vol_ratio_5_20",
    "vol_ratio_10_20",
    "vol_momentum",
    "vol_spike",
    "return_5d",
    "ret_1",
    "ret_3",
    "ret_5",
    "trend_strength",
    "price_vs_ma",
    "volume_change",
    "vol_compression",
    "vol_change",
]

TARGETS = [
    "vol_regime",
    "vol_direction",
    "price_direction",
    "forward_return_1d",
]


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
    close  = df["Close"]
    volume = df["Volume"]

    log_ret = np.log(close / close.shift(1))

    df["log_return"]      = log_ret
    df["abs_return"]      = log_ret.abs()
    df["ret_1"]           = close.pct_change(1)
    df["ret_3"]           = close.pct_change(3)
    df["ret_5"]           = close.pct_change(5)
    df["return_5d"]       = log_ret.rolling(5).sum()
    df["vol_5"]           = log_ret.rolling(5).std()
    df["vol_10"]          = log_ret.rolling(10).std()
    df["vol_20"]          = log_ret.rolling(20).std()
    df["vol_ratio_5_20"]  = df["vol_5"]  / (df["vol_20"] + 1e-8)
    df["vol_ratio_10_20"] = df["vol_10"] / (df["vol_20"] + 1e-8)
    df["vol_momentum"]    = df["vol_5"]  - df["vol_10"]
    df["vol_spike"]       = (df["vol_5"] > df["vol_20"] * 1.5).astype(int)
    df["trend_strength"]  = df["ret_5"].rolling(5).mean()
    df["price_vs_ma"]     = close / (close.rolling(20).mean() + 1e-8)
    df["volume_change"]   = np.log1p(volume.pct_change())
    df["vol_compression"] = df["vol_20"].rolling(10).min() / (df["vol_20"] + 1e-8)
    df["vol_change"]      = df["vol_5"] / (df["vol_5"].shift(1) + 1e-8)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df[FEATURES] = df[FEATURES].clip(-10, 10)
    df.dropna(inplace=True)
    print(f"    {len(FEATURES)} features — {len(df)} clean rows")
    return df


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    print("[+] Computing targets...")

    log_ret = df["log_return"]

    # vol_regime: is current vol above its rolling median? (no lookahead)
    rolling_median   = df["vol_5"].rolling(20, min_periods=10).median()
    df["vol_regime"] = (df["vol_5"] > rolling_median).astype(int)

    # vol_direction: will vol_5 increase tomorrow?
    df["vol_direction"] = (df["vol_5"].shift(-1) > df["vol_5"]).astype(int)

    # price_direction: will price go up tomorrow?
    # forward_return_1d is the actual return — price_direction is its sign
    df["forward_return_1d"] = log_ret.shift(-1)
    df["price_direction"]   = (df["forward_return_1d"] > 0).astype(int)

    before = len(df)
    df.dropna(subset=["vol_direction", "price_direction", "forward_return_1d"], inplace=True)
    print(f"    {before - len(df)} rows dropped (lookahead window)")
    print(f"    vol_direction   — up: {df['vol_direction'].mean()*100:.1f}%")
    print(f"    price_direction — up: {df['price_direction'].mean()*100:.1f}%")
    return df


def save_features(df: pd.DataFrame, ticker: str) -> str:
    path = os.path.join(DATA_DIR, f"{ticker.lower()}_features.parquet")
    df.to_parquet(path)
    print(f"[+] Saved → {path}")
    return path


def run_pipeline(ticker: str) -> pd.DataFrame:
    df = fetch_data(ticker)
    df = add_indicators(df)
    df = add_targets(df)
    save_features(df, ticker)
    print(f"[✓] Pipeline done for {ticker} — {len(df)} rows\n")
    return df


if __name__ == "__main__":
    for ticker in ["^GSPC", "AAPL"]:
        df = run_pipeline(ticker)
        print(df[FEATURES + TARGETS].tail(3))
        print()