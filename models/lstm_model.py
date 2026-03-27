import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import RobustScaler
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data.data_pipeline import FEATURES

CONTEXT_LEN = 60
PRED_LEN     = 5
HIDDEN_SIZE  = 256
NUM_LAYERS   = 3
DROPOUT      = 0.3

class StockLSTM(nn.Module):
    def __init__(self, input_size=len(FEATURES), hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=DROPOUT
        )
        self.norm    = nn.LayerNorm(hidden_size)
        self.fc1     = nn.Linear(hidden_size, 128)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(DROPOUT)
        self.fc2     = nn.Linear(128, 64)
        self.fc3     = nn.Linear(64, PRED_LEN)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last = self.norm(lstm_out[:, -1, :])
        out  = self.fc1(last)
        out  = self.relu(out)
        out  = self.dropout(out)
        out  = self.fc2(out)
        out  = self.relu(out)
        out  = self.fc3(out)

        # FIX 3: Bound output to ±5% max daily move using tanh
        # tanh outputs (-1, 1), scaled to (-0.05, 0.05)
        out = torch.tanh(out) * 0.05
        return out


def add_log_returns(df):
    """
    FIX 1: Add log return column as prediction target.
    log_return = log(Close_t / Close_{t-1})
    This makes target scale-invariant — works regardless of price level.
    """
    import numpy as np
    df = df.copy()
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df.dropna(inplace=True)
    return df


def create_sequences(df, context_len=CONTEXT_LEN, pred_len=PRED_LEN):
    """
    FIX 1: y is now log returns (next pred_len days), not absolute prices.
    This makes evaluation scale-invariant and prevents explosion.
    """
    feat_data    = df[FEATURES].values
    returns_data = df["log_return"].values

    X, y = [], []
    for i in range(len(feat_data) - context_len - pred_len + 1):
        X.append(feat_data[i : i + context_len])
        y.append(returns_data[i + context_len : i + context_len + pred_len])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def fit_scaler(df):
    # FIX 2: RobustScaler fitted only on features (not returns)
    scaler = RobustScaler()
    scaler.fit(df[FEATURES].values)
    return scaler


def scale_df(df, scaler):
    df_scaled          = df.copy()
    df_scaled[FEATURES] = scaler.transform(df[FEATURES].values)
    return df_scaled


if __name__ == "__main__":
    model = StockLSTM()
    print(model)
    dummy = torch.randn(8, CONTEXT_LEN, len(FEATURES))
    out   = model(dummy)
    print(f"\nInput : {dummy.shape}")
    print(f"Output: {out.shape}")
    print(f"Output range (should be within ±0.05): min={out.min():.4f} max={out.max():.4f}")