import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD
import os
 
# conifg
START_DATE = "2004-08-19"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "feature_store")
os.makedirs(DATA_DIR, exist_ok=True)
 
 
#  fetchin OHLCV 
def fetch_data(ticker: str, start: str = START_DATE) -> pd.DataFrame:
    print(f"[+] Fetching data for {ticker}...")
    df = yf.download(ticker, start=start, auto_adjust=True)
 
    if df.empty:
        raise ValueError(f"No data found for ticker: {ticker}")
 
    # flatten multi-level columns 
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
 
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.dropna(inplace=True)
 
    print(f"    {len(df)} rows fetched ({df.index[0].date()} to {df.index[-1].date()})")
    return df
 
 
# technical indicators
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    print("[+] Calculating technical indicators...")
 
    # RSI (14-day)
    rsi = RSIIndicator(close=df["Close"], window=14)
    df["RSI14"] = rsi.rsi()
 
    # MACD
    macd = MACD(close=df["Close"])
    df["MACD"] = macd.macd()
 
    df.dropna(inplace=True)
    print(f"    Features: {list(df.columns)}")
    return df
 
 
# save
def save_features(df: pd.DataFrame, ticker: str):
    path = os.path.join(DATA_DIR, f"{ticker.lower()}_features.parquet")
    df.to_parquet(path)
    print(f"[+] Saved to {path}")
    return path
 
 
# main pipeline shit
def run_pipeline(ticker: str) -> pd.DataFrame:
    df = fetch_data(ticker)
    df = add_indicators(df)
    save_features(df, ticker)
    print(f"[✓] Pipeline complete for {ticker}\n")
    return df
 
 
# test
if __name__ == "__main__":
    # Test with S&P 500 (parent) and AAPL (child)
    for ticker in ["^GSPC", "AAPL"]:
        df = run_pipeline(ticker)
        print(df.tail(3))
        print()
 