import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import RobustScaler
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data.data_pipeline import FEATURES

CONTEXT_LEN = 30
HIDDEN_SIZE = 64
NUM_LAYERS  = 1


class StockLSTM(nn.Module):
    """
    Two-head LSTM:
      - vol_head:    predicts realized_vol_5d (regression, Softplus output)
      - regime_head: predicts trend_regime probability (classification, Sigmoid output)

    Why two heads on one shared LSTM:
      The LSTM learns a shared market representation. Vol and regime
      are related — trending markets tend to have lower vol, turbulent
      markets higher vol. Sharing the LSTM layers lets both heads
      benefit from the same learned patterns.
    """
    def __init__(self, input_size=len(FEATURES), hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(hidden_size)

        # Shared dense layer — both heads read from this
        self.shared = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
        )

        # Head 1: volatility regression
        # Softplus ensures output is always positive (vol can't be negative)
        self.vol_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Softplus()
        )

        # Head 2: regime classification
        # Sigmoid outputs probability in (0, 1)
        # > 0.65 = confident uptrend, < 0.35 = confident downtrend, else flat
        self.regime_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last   = self.norm(lstm_out[:, -1, :])
        shared = self.shared(last)
        vol    = self.vol_head(shared)      # shape: (batch, 1)
        regime = self.regime_head(shared)   # shape: (batch, 1)
        return vol, regime


def create_sequences(df, context_len=CONTEXT_LEN):
    """
    Build input sequences from feature columns.
    Targets are realized_vol_5d and trend_regime — both must exist in df.

    X shape:     (n_samples, context_len, n_features)
    y_vol shape: (n_samples, 1)
    y_reg shape: (n_samples, 1)
    """
    feat    = df[FEATURES].values
    y_vol   = df["realized_vol_5d"].values
    y_reg   = df["trend_regime"].values

    X, yv, yr = [], [], []
    for i in range(context_len, len(feat)):
        X.append(feat[i - context_len:i])
        yv.append(y_vol[i])
        yr.append(y_reg[i])

    return (
        np.array(X,  dtype=np.float32),
        np.array(yv, dtype=np.float32).reshape(-1, 1),
        np.array(yr, dtype=np.float32).reshape(-1, 1),
    )


def fit_scaler(df):
    """Fit RobustScaler on feature columns only — never on targets."""
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
    dummy        = torch.randn(8, CONTEXT_LEN, len(FEATURES))
    vol, regime  = model(dummy)
    print(f"\nInput  : {dummy.shape}")
    print(f"Vol out: {vol.shape}    (predicted volatility, always positive)")
    print(f"Regime : {regime.shape} (P(uptrend), range 0-1)")
    print(f"Params : {sum(p.numel() for p in model.parameters()):,}")