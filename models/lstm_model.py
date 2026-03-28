import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import RobustScaler
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data.data_pipeline import FEATURES

CONTEXT_LEN = 30    # reduced from 60 — less noise, more signal
PRED_LEN    = 5
HIDDEN_SIZE = 64    # reduced from 256 — prevents overfit on small dataset
NUM_LAYERS  = 1     # reduced from 3 — simpler model generalizes better
DROPOUT     = 0.0   # no dropout with 1 layer (causes issues)


class StockLSTM(nn.Module):
    def __init__(self, input_size=len(FEATURES), hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.norm    = nn.LayerNorm(hidden_size)
        self.fc1     = nn.Linear(hidden_size, 32)
        self.relu    = nn.ReLU()
        self.fc2     = nn.Linear(32, PRED_LEN)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last = self.norm(lstm_out[:, -1, :])
        out  = self.fc1(last)
        out  = self.relu(out)
        out  = self.fc2(out)
        return out


def add_log_returns(df):
    df = df.copy()
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df.dropna(inplace=True)
    return df


def create_sequences(df, context_len=CONTEXT_LEN, pred_len=PRED_LEN):
    feat_data    = df[FEATURES].values
    returns_data = df["log_return"].values
    X, y = [], []
    for i in range(len(feat_data) - context_len - pred_len + 1):
        X.append(feat_data[i : i + context_len])
        y.append(returns_data[i + context_len : i + context_len + pred_len])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def fit_scaler(df):
    scaler = RobustScaler()
    scaler.fit(df[FEATURES].values)
    return scaler


def scale_df(df, scaler):
    df_scaled           = df.copy()
    df_scaled[FEATURES] = scaler.transform(df[FEATURES].values)
    return df_scaled


if __name__ == "__main__":
    model = StockLSTM()
    print(model)
    dummy = torch.randn(8, CONTEXT_LEN, len(FEATURES))
    out   = model(dummy)
    print(f"\nInput : {dummy.shape}")
    print(f"Output: {out.shape}")
    print(f"Param count: {sum(p.numel() for p in model.parameters()):,}")