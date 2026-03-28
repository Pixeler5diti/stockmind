import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import os
import sys
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler
import mlflow
import mlflow.pytorch
from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models.lstm_model import StockLSTM, create_sequences, scale_df, add_log_returns, CONTEXT_LEN, PRED_LEN, HIDDEN_SIZE, NUM_LAYERS, DROPOUT
from data.data_pipeline import run_pipeline, FEATURES

BATCH_SIZE    = 16
PARENT_EPOCHS = 100
CHILD_EPOCHS  = 50
LEARNING_RATE = 0.001
FINE_TUNE_LR  = 0.0005
PATIENCE      = 15
GRAD_CLIP     = 0.5
OUTPUTS_DIR   = os.path.join(os.path.dirname(__file__), "..", "outputs")
FEATURE_STORE = os.path.join(os.path.dirname(__file__), "..", "feature_store")

# ── MLflow setup ───────────────────────────────────────
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("stckmind")


def load_parquet(ticker: str) -> pd.DataFrame:
    path = os.path.join(FEATURE_STORE, f"{ticker.lower()}_features.parquet")
    if not os.path.exists(path):
        return run_pipeline(ticker)
    return pd.read_parquet(path)


def get_output_dir(ticker: str) -> str:
    path = os.path.join(OUTPUTS_DIR, ticker.lower())
    os.makedirs(path, exist_ok=True)
    return path


def prepare_data(df):
    df = add_log_returns(df)

    split_row = int(len(df) * 0.8)
    train_df  = df.iloc[:split_row]
    scaler    = RobustScaler()
    scaler.fit(train_df[FEATURES].values)

    df_scaled           = df.copy()
    df_scaled[FEATURES] = scaler.transform(df[FEATURES].values)

    X, y  = create_sequences(df_scaled)
    split = int(len(X) * 0.8)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"    Train sequences: {len(X_train)} | Test sequences: {len(X_test)}")
    print(f"    Target mean: {y_train.mean():.6f} | Target std: {y_train.std():.6f}")

    train_ds     = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    return train_loader, X_test, y_test, scaler


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
        mlflow.log_metric(f"{label}_train_loss", avg_loss, step=epoch)
        print(f"  [{label}] Epoch {epoch:03d}/{epochs} — Loss: {avg_loss:.7f}", end="")

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


def evaluate(model, X_test, y_test, label=""):
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_test)).numpy()

    mae        = np.mean(np.abs(preds - y_test))
    rmse       = np.sqrt(np.mean((preds - y_test) ** 2))
    naive_rmse = np.sqrt(np.mean(y_test ** 2))
    ss_res     = np.sum((y_test - preds) ** 2)
    ss_tot     = np.sum((y_test - np.mean(y_test)) ** 2)
    r2         = 1 - ss_res / ss_tot
    dir_acc    = np.mean(np.sign(preds) == np.sign(y_test)) * 100
    pred_std   = preds.std()

    print(f"\n  Evaluation:")
    print(f"    MAE            : {mae:.7f}")
    print(f"    RMSE           : {rmse:.7f}")
    print(f"    Naive RMSE     : {naive_rmse:.7f}")
    print(f"    R²             : {r2:.4f}")
    print(f"    Direction acc  : {dir_acc:.1f}%  (50% = random)")
    print(f"    Model vs Naive : {'BETTER ✓' if rmse < naive_rmse else 'WORSE ✗'}")
    print(f"    Pred std       : {pred_std:.7f}  (0 = collapsed)")

    metrics = {
        "mae": mae, "rmse": rmse, "naive_rmse": naive_rmse,
        "r2": r2, "dir_acc": dir_acc, "pred_std": pred_std
    }

    # Log to MLflow
    for k, v in metrics.items():
        mlflow.log_metric(f"{label}_{k}", v)

    return metrics


def train_parent(ticker="^GSPC"):
    print(f"\n{'='*55}")
    print(f"  TRAINING PARENT MODEL — {ticker}")
    print(f"{'='*55}")

    df = load_parquet(ticker)
    print(f"  [+] {len(df)} rows loaded")

    train_loader, X_test, y_test, scaler = prepare_data(df)

    with mlflow.start_run(run_name=f"parent_{ticker.lower().replace('^','')}"):
        # Log hyperparameters
        mlflow.log_params({
            "ticker":       ticker,
            "model_type":   "parent",
            "hidden_size":  HIDDEN_SIZE,
            "num_layers":   NUM_LAYERS,
            "dropout":      DROPOUT,
            "context_len":  CONTEXT_LEN,
            "pred_len":     PRED_LEN,
            "batch_size":   BATCH_SIZE,
            "epochs":       PARENT_EPOCHS,
            "lr":           LEARNING_RATE,
            "n_features":   len(FEATURES),
            "optimizer":    "Adam",
            "loss":         "MSELoss",
        })

        model     = StockLSTM(input_size=len(FEATURES))
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.MSELoss()

        model   = train_loop(model, train_loader, optimizer, criterion, PARENT_EPOCHS, label="parent")
        metrics = evaluate(model, X_test, y_test, label="parent")

        out_dir = get_output_dir(ticker)
        model_path  = os.path.join(out_dir, "parent_model.pt")
        scaler_path = os.path.join(out_dir, "parent_scaler.pkl")
        torch.save(model.state_dict(), model_path)
        joblib.dump(scaler, scaler_path)

        # Log artifacts to MLflow
        mlflow.log_artifact(model_path)
        mlflow.log_artifact(scaler_path)
        mlflow.set_tag("status", "completed")

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
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LEARNING_RATE
        )
        print("  [+] LSTM frozen — training head only")
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=FINE_TUNE_LR)
        print("  [+] Fine-tuning all layers")

    df = load_parquet(ticker)
    print(f"  [+] {len(df)} rows loaded")

    train_loader, X_test, y_test, scaler = prepare_data(df)

    with mlflow.start_run(run_name=f"child_{ticker.lower()}_{strategy}"):
        mlflow.log_params({
            "ticker":          ticker,
            "model_type":      "child",
            "strategy":        strategy,
            "hidden_size":     HIDDEN_SIZE,
            "num_layers":      NUM_LAYERS,
            "dropout":         DROPOUT,
            "context_len":     CONTEXT_LEN,
            "pred_len":        PRED_LEN,
            "batch_size":      BATCH_SIZE,
            "epochs":          CHILD_EPOCHS,
            "lr":              FINE_TUNE_LR if strategy == "fine_tune" else LEARNING_RATE,
            "n_features":      len(FEATURES),
            "optimizer":       "Adam",
            "loss":            "MSELoss",
            "parent_ticker":   "^GSPC",
        })

        criterion = nn.MSELoss()
        model     = train_loop(model, train_loader, optimizer, criterion, CHILD_EPOCHS, label=ticker)
        metrics   = evaluate(model, X_test, y_test, label=ticker)

        out_dir     = get_output_dir(ticker)
        model_path  = os.path.join(out_dir, f"{ticker.lower()}_child_model.pt")
        scaler_path = os.path.join(out_dir, f"{ticker.lower()}_scaler.pkl")
        torch.save(model.state_dict(), model_path)
        joblib.dump(scaler, scaler_path)

        mlflow.log_artifact(model_path)
        mlflow.log_artifact(scaler_path)
        mlflow.set_tag("status", "completed")

    print(f"  [✓] Child model saved\n")
    return model, scaler, metrics


if __name__ == "__main__":
    train_parent("^GSPC")
    train_child("AAPL", strategy="fine_tune")