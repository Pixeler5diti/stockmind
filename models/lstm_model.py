import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# ── Config ─────────────────────────────────────────────
FEATURES = ["Open", "High", "Low", "Close", "Volume", "RSI14", "MACD"]
CONTEXT_LEN = 60      # days of history as input
PRED_LEN = 5          # days to predict
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.2


# ── LSTM Architecture ──────────────────────────────────
class StockLSTM(nn.Module):
    def __init__(self, input_size=7, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=DROPOUT
        )

        # Dense layers
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(DROPOUT)
        self.fc2 = nn.Linear(64, PRED_LEN)   # output: 5 future prices

    def forward(self, x):
        # x shape: (batch, seq_len=60, features=7)
        lstm_out, _ = self.lstm(x)
        last = lstm_out[:, -1, :]   # take last timestep
        out = self.fc1(last)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out                  # shape: (batch, 5)


# ── Sequence Builder ───────────────────────────────────
def create_sequences(df, context_len=CONTEXT_LEN, pred_len=PRED_LEN):
    """
    Slices dataframe into overlapping windows.
    X: (n_samples, context_len, n_features)
    y: (n_samples, pred_len)  — Close prices only
    """
    data = df[FEATURES].values
    close_idx = FEATURES.index("Close")

    X, y = [], []
    for i in range(len(data) - context_len - pred_len + 1):
        X.append(data[i : i + context_len])
        y.append(data[i + context_len : i + context_len + pred_len, close_idx])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ── Scaler ─────────────────────────────────────────────
def fit_scaler(df):
    scaler = MinMaxScaler()
    scaler.fit(df[FEATURES].values)
    return scaler


def scale_df(df, scaler):
    scaled = scaler.transform(df[FEATURES].values)
    df_scaled = df.copy()
    df_scaled[FEATURES] = scaled
    return df_scaled


# ── Quick sanity check ─────────────────────────────────
if __name__ == "__main__":
    model = StockLSTM()
    print(model)

    # Dummy forward pass
    dummy = torch.randn(8, CONTEXT_LEN, len(FEATURES))   # batch=8
    out = model(dummy)
    print(f"\nInput shape : {dummy.shape}")
    print(f"Output shape: {out.shape}")   # should be (8, 5)