import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import os
import sys
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models.lstm_model import StockLSTM, create_sequences, scale_df, add_log_returns, CONTEXT_LEN, PRED_LEN
from data.data_pipeline import run_pipeline, FEATURES

BATCH_SIZE    = 32
PARENT_EPOCHS = 50
CHILD_EPOCHS  = 50
LEARNING_RATE = 0.001
FINE_TUNE_LR  = 0.0001
PATIENCE      = 8
GRAD_CLIP     = 1.0
OUTPUTS_DIR   = os.path.join(os.path.dirname(__file__), "..", "outputs")
FEATURE_STORE = os.path.join(os.path.dirname(__file__), "..", "feature_store")


def load_parquet(ticker: str) -> pd.DataFrame:
    path = os.path.join(FEATURE_STORE, f"{ticker.lower()}_features.parquet")
    if not os.path.exists(path):
        print(f"[!] Data not found for {ticker}, running pipeline...")
        return run_pipeline(ticker)
    return pd.read_parquet(path)


def get_output_dir(ticker: str) -> str:
    path = os.path.join(OUTPUTS_DIR, ticker.lower())
    os.makedirs(path, exist_ok=True)
    return path


def prepare_data(df):
    """
    FIX 1+2: 
    - Add log returns as target
    - Fit scaler ONLY on train portion (80%)
    - Walk-forward split on sequences
    """
    df = add_log_returns(df)

    # FIX 2: fit scaler on train rows only (first 80% of raw data)
    split_row = int(len(df) * 0.8)
    train_df  = df.iloc[:split_row]
    scaler    = RobustScaler()
    scaler.fit(train_df[FEATURES].values)

    # Scale full df using train scaler
    df_scaled          = df.copy()
    df_scaled[FEATURES] = scaler.transform(df[FEATURES].values)

    # Create sequences from scaled features + log return targets
    X, y = create_sequences(df_scaled)

    # Walk-forward split on sequences
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Save last close of train period for inference reconstruction
    last_train_close = float(train_df["Close"].iloc[-1])

    print(f"    Train sequences: {len(X_train)} | Test sequences: {len(X_test)}")

    train_ds     = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    return train_loader, X_test, y_test, scaler, df


def train_loop(model, train_loader, optimizer, criterion, epochs, label=""):
    best_loss    = float("inf")
    patience_cnt = 0
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss  = criterion(preds, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"  [{label}] Epoch {epoch:02d}/{epochs} — Loss: {avg_loss:.6f}", end="")

        if avg_loss < best_loss:
            best_loss    = avg_loss
            patience_cnt = 0
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            print(" ✓")
        else:
            patience_cnt += 1
            print(f" (patience {patience_cnt}/{PATIENCE})")
            if patience_cnt >= PATIENCE:
                print(f"  [!] Early stopping at epoch {epoch}")
                break

    if best_weights:
        model.load_state_dict(best_weights)
    return model


def evaluate(model, X_test, y_test):
    """
    FIX 5: Evaluate on log returns directly (no inverse scaling needed).
    Add MAPE and naive baseline comparison.
    y_test shape: (n, PRED_LEN) — log returns
    """
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_test)).numpy()

    # Both preds and y_test are log returns — same scale, no inverse needed
    mae  = np.mean(np.abs(preds - y_test))
    mse  = np.mean((preds - y_test) ** 2)
    rmse = np.sqrt(mse)

    # MAPE on returns
    mask = np.abs(y_test) > 1e-6   # avoid div by zero on flat days
    mape = np.mean(np.abs((preds[mask] - y_test[mask]) / y_test[mask])) * 100

    # R² on log returns
    ss_res = np.sum((y_test - preds) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2     = 1 - ss_res / ss_tot

    # Naive baseline: predict 0 return (price stays flat)
    naive_mse  = np.mean(y_test ** 2)
    naive_rmse = np.sqrt(naive_mse)

    print(f"\n  Evaluation (log returns):")
    print(f"    MAE   : {mae:.6f}  (log return units)")
    print(f"    RMSE  : {rmse:.6f}")
    print(f"    MAPE  : {mape:.2f}%")
    print(f"    R²    : {r2:.4f}")
    print(f"    Naive RMSE (flat baseline): {naive_rmse:.6f}")
    print(f"    Model vs Naive: {'BETTER ✓' if rmse < naive_rmse else 'WORSE ✗'}")

    return {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2, "naive_rmse": naive_rmse}


def train_parent(ticker="^GSPC"):
    print(f"\n{'='*55}")
    print(f"  TRAINING PARENT MODEL — {ticker}")
    print(f"{'='*55}")

    df = load_parquet(ticker)
    print(f"  [+] {len(df)} rows loaded")

    train_loader, X_test, y_test, scaler, df_full = prepare_data(df)

    model     = StockLSTM(input_size=len(FEATURES))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.HuberLoss()

    model   = train_loop(model, train_loader, optimizer, criterion, PARENT_EPOCHS, label="Parent")
    metrics = evaluate(model, X_test, y_test)

    out_dir = get_output_dir(ticker)
    torch.save(model.state_dict(), os.path.join(out_dir, "parent_model.pt"))
    joblib.dump(scaler, os.path.join(out_dir, "parent_scaler.pkl"))
    print(f"  [✓] Parent model saved\n")
    return model, scaler, metrics


def train_child(ticker: str, strategy: str = "fine_tune"):
    print(f"\n{'='*55}")
    print(f"  TRAINING CHILD MODEL — {ticker} [{strategy}]")
    print(f"{'='*55}")

    parent_model_path = os.path.join(get_output_dir("^GSPC"), "parent_model.pt")
    if not os.path.exists(parent_model_path):
        print("[!] Parent model not found — training parent first...")
        train_parent()

    model = StockLSTM(input_size=len(FEATURES))
    model.load_state_dict(torch.load(parent_model_path, weights_only=True))
    print(f"  [+] Loaded parent weights")

    if strategy == "freeze":
        for param in model.lstm.parameters():
            param.requires_grad = False
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LEARNING_RATE, weight_decay=1e-4
        )
        print("  [+] LSTM frozen — training head only")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=FINE_TUNE_LR, weight_decay=1e-4)
        print("  [+] Fine-tuning all layers")

    df = load_parquet(ticker)
    print(f"  [+] {len(df)} rows loaded")

    train_loader, X_test, y_test, scaler, df_full = prepare_data(df)
    criterion = nn.HuberLoss()

    model   = train_loop(model, train_loader, optimizer, criterion, CHILD_EPOCHS, label=ticker)
    metrics = evaluate(model, X_test, y_test)

    out_dir = get_output_dir(ticker)
    torch.save(model.state_dict(), os.path.join(out_dir, f"{ticker.lower()}_child_model.pt"))
    joblib.dump(scaler, os.path.join(out_dir, f"{ticker.lower()}_scaler.pkl"))
    print(f"  [✓] Child model saved\n")
    return model, scaler, metrics


if __name__ == "__main__":
    train_parent("^GSPC")
    train_child("AAPL", strategy="fine_tune")