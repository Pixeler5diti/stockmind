import torch
import torch.nn as nn
import numpy as np

CONTEXT_LEN = 20
HIDDEN_SIZE = 48
NUM_LAYERS  = 1


class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=HIDDEN_SIZE):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=NUM_LAYERS,
            batch_first=True,
        )
        self.norm    = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.2)
        self.fc1     = nn.Linear(hidden_size, 24)
        self.relu    = nn.ReLU()
        self.fc2     = nn.Linear(24, 1)
        # No sigmoid here — BCEWithLogitsLoss handles it during training
        # torch.sigmoid() applied manually at inference

    def forward(self, x):
        out, _ = self.lstm(x)
        last   = self.norm(out[:, -1, :])
        last   = self.dropout(last)
        last   = self.fc1(last)
        last   = self.relu(last)
        return self.fc2(last)   # raw logit


def create_sequences(df, feature_cols, context_len=CONTEXT_LEN):
    feat = df[feature_cols].values
    y    = df["vol_direction"].values
    X, labels = [], []
    for i in range(context_len, len(feat)):
        X.append(feat[i - context_len:i])
        labels.append(y[i])
    return (
        np.array(X,      dtype=np.float32),
        np.array(labels, dtype=np.float32).reshape(-1, 1),
    )


if __name__ == "__main__":
    model = StockLSTM(input_size=10)
    print(model)
    dummy = torch.randn(8, CONTEXT_LEN, 10)
    out   = model(dummy)
    print(f"Input : {dummy.shape}")
    print(f"Output: {out.shape}  (raw logit — apply sigmoid for probability)")