import yfinance as yf
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

START_DATE = "2020-01-01"
DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "feature_store")
os.makedirs(DATA_DIR, exist_ok=True)


# -------------------------
# FEATURES
# -------------------------
FEATURES = [
    "log_return",
    "abs_return",

    "vol_5",
    "vol_10",
    "vol_20",
    "vol_ratio_5_20",
    "vol_ratio_10_20",
    "vol_ratio",
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
]

TARGETS = [
    "vol_regime",
    "vol_direction",
    "forward_return_1d",
]


# -------------------------
# FETCH DATA
# -------------------------
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


# -------------------------
# FEATURE ENGINEERING
# -------------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    print("[+] Engineering features...")

    close  = df["Close"]
    volume = df["Volume"]

    # RETURNS
    log_ret = np.log(close / close.shift(1))
    df["log_return"] = log_ret
    df["abs_return"] = log_ret.abs()

    df["ret_1"] = close.pct_change(1)
    df["ret_3"] = close.pct_change(3)
    df["ret_5"] = close.pct_change(5)

    df["return_5d"] = log_ret.rolling(5).sum()

    # VOLATILITY
    df["vol_5"]  = log_ret.rolling(5).std()
    df["vol_10"] = log_ret.rolling(10).std()
    df["vol_20"] = log_ret.rolling(20).std()

    df["vol_ratio_5_20"]  = df["vol_5"]  / (df["vol_20"] + 1e-8)
    df["vol_ratio_10_20"] = df["vol_10"] / (df["vol_20"] + 1e-8)

    df["vol_ratio"]    = df["vol_5"] / (df["vol_20"] + 1e-8)
    df["vol_momentum"] = df["vol_5"] - df["vol_10"]

    df["vol_spike"] = (df["vol_5"] > df["vol_20"] * 1.5).astype(int)

    # TREND
    df["trend_strength"] = df["ret_5"].rolling(5).mean()
    df["price_vs_ma"] = close / (close.rolling(20).mean() + 1e-8)

    # VOLUME
    df["volume_change"] = np.log1p(volume.pct_change())

    # VOL COMPRESSION
    df["vol_compression"] = df["vol_20"].rolling(10).min() / (df["vol_20"] + 1e-8)

    # CLEANUP
    df.dropna(inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # SAFE CLIP (no missing column crash)
    safe_features = [f for f in FEATURES if f in df.columns]
    df[safe_features] = df[safe_features].clip(-10, 10)

    print(f"    {len(safe_features)} features — {len(df)} clean rows")
    return df


# -------------------------
# TARGETS
# -------------------------
def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    print("[+] Computing targets...")

    rolling_median = df["vol_5"].rolling(20, min_periods=10).median()
    df["vol_regime"] = (df["vol_5"] > rolling_median).astype(int)

    df["vol_direction"] = (df["vol_5"].shift(-1) > df["vol_5"]).astype(int)

    df["forward_return_1d"] = df["log_return"].shift(-1)

    df.dropna(subset=["vol_direction", "forward_return_1d"], inplace=True)

    print(f"    vol_regime  — high: {df['vol_regime'].mean()*100:.1f}%")
    print(f"    vol_direction — up: {df['vol_direction'].mean()*100:.1f}%")

    return df


# -------------------------
# MARKET CONTEXT
# -------------------------
def add_market_context(df: pd.DataFrame) -> pd.DataFrame:
    print("[+] Adding market context (S&P500)...")

    sp500_path = os.path.join(DATA_DIR, "^gspc_features.parquet")

    if not os.path.exists(sp500_path):
        sp500_df = run_pipeline("^GSPC")
    else:
        sp500_df = pd.read_parquet(sp500_path)

    sp500_df = sp500_df[["vol_direction"]].copy()
    sp500_df.rename(columns={"vol_direction": "sp500_vol_dir"}, inplace=True)

    df = df.merge(
        sp500_df,
        left_index=True,
        right_index=True,
        how="left"
    )

    df["sp500_vol_dir"].fillna(0, inplace=True)

    return df


# -------------------------
# SAVE
# -------------------------
def save_features(df: pd.DataFrame, ticker: str) -> str:
    path = os.path.join(DATA_DIR, f"{ticker.lower()}_features.parquet")
    df.to_parquet(path)
    print(f"[+] Saved → {path}")
    return path


# -------------------------
# PIPELINE
# -------------------------
def run_pipeline(ticker: str) -> pd.DataFrame:
    df = fetch_data(ticker)
    df = add_indicators(df)
    df = add_targets(df)

    

        # clip again after adding sp500 feature
    safe_features = [f for f in FEATURES if f in df.columns]
    df[safe_features] = df[safe_features].clip(-10, 10)

    save_features(df, ticker)

    print(f"[✓] Pipeline done for {ticker} — {len(df)} rows\n")
    return df


# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    for ticker in ["^GSPC", "AAPL"]:
        df = run_pipeline(ticker)

        # SAFE PRINT (no crash ever)
        safe_cols = [c for c in (FEATURES + TARGETS) if c in df.columns]
        print(df[safe_cols].tail(3))
        print()