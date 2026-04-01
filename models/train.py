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
from models.lstm_model import StockLSTM, create_sequences, scale_df, fit_scaler, CONTEXT_LEN, HIDDEN_SIZE, NUM_LAYERS
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
    """
    Scaler is fit on first 80% of raw rows (train period only).
    Sequences are then built from the full scaled df.
    Walk-forward split: test = last 20% of sequences.
    """
    split_row = int(len(df) * 0.8)
    train_df  = df.iloc[:split_row]

    # FIT scaler on train rows only — no leakage
    scaler = RobustScaler()
    scaler.fit(train_df[FEATURES].values)

    df_scaled           = df.copy()
    df_scaled[FEATURES] = scaler.transform(df[FEATURES].values)

    # create_sequences now returns X, y_vol, y_reg
    X, y_vol, y_reg = create_sequences(df_scaled)

    split = int(len(X) * 0.8)
    X_train, X_test     = X[:split],     X[split:]
    yv_train, yv_test   = y_vol[:split], y_vol[split:]
    yr_train, yr_test   = y_reg[:split], y_reg[split:]

    print(f"    Train sequences : {len(X_train)} | Test sequences: {len(X_test)}")
    print(f"    Vol target      — mean: {yv_train.mean():.6f}  std: {yv_train.std():.6f}")
    print(f"    Regime target   — uptrend: {yr_train.mean()*100:.1f}%")

    train_ds = TensorDataset(
        torch.tensor(X_train),
        torch.tensor(yv_train),
        torch.tensor(yr_train),
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    return train_loader, X_test, yv_test, yr_test, scaler


def train_loop(model, train_loader, optimizer, epochs, label=""):
    """
    Two-head training loop.
    Loss = MSE on vol prediction + 0.5 * BCE on regime classification.
    Vol loss weighted higher because volatility prediction is the primary signal.
    """
    vol_loss_fn = nn.MSELoss()
    reg_loss_fn = nn.BCELoss()

    best_loss    = float("inf")
    patience_cnt = 0
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0

        for X_batch, yv_batch, yr_batch in train_loader:
            optimizer.zero_grad()

            pred_vol, pred_reg = model(X_batch)

            loss_vol = vol_loss_fn(pred_vol, yv_batch)
            loss_reg = reg_loss_fn(pred_reg, yr_batch)

            # Combined loss — vol weighted 1.0, regime weighted 0.5
            loss = loss_vol + 0.5 * loss_reg

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            total_vol_loss += loss_vol.item()
            total_reg_loss += loss_reg.item()

        avg_loss = total_loss / len(train_loader)
        mlflow.log_metric(f"{label}_train_loss", avg_loss, step=epoch)
        print(f"  [{label}] Epoch {epoch:03d}/{epochs} — Loss: {avg_loss:.7f}", end="")

        mlflow.log_metric(f"{label}_vol_loss",   avg_vol,   step=epoch)
        mlflow.log_metric(f"{label}_reg_loss",   avg_reg,   step=epoch)
        mlflow.log_metric(f"{label}_total_loss", avg_total, step=epoch)

        print(f"  [{label}] Epoch {epoch:03d}/{epochs} — "
              f"Vol: {avg_vol:.5f}  Reg: {avg_reg:.5f}  Total: {avg_total:.5f}", end="")

        if avg_total < best_loss:
            best_loss    = avg_total
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


def evaluate(model, X_test, yv_test, yr_test, label=""):
    """
    Evaluate both heads separately.

    Vol head  — R², RMSE vs naive baseline (predict mean vol every day)
    Regime head — accuracy, precision for uptrend class
    """
    model.eval()
    with torch.no_grad():
        pred_vol, pred_reg = model(torch.tensor(X_test))

    pred_vol = pred_vol.numpy().flatten()
    pred_reg = pred_reg.numpy().flatten()
    yv_test  = yv_test.flatten()
    yr_test  = yr_test.flatten()

    # Vol metrics
    vol_mae       = np.mean(np.abs(pred_vol - yv_test))
    vol_rmse      = np.sqrt(np.mean((pred_vol - yv_test) ** 2))
    naive_vol_rmse= np.sqrt(np.mean((yv_test - yv_test.mean()) ** 2))
    ss_res        = np.sum((yv_test - pred_vol) ** 2)
    ss_tot        = np.sum((yv_test - yv_test.mean()) ** 2)
    vol_r2        = 1 - ss_res / ss_tot

    # Regime metrics
    reg_pred_binary = (pred_reg > 0.5).astype(int)
    reg_accuracy    = np.mean(reg_pred_binary == yr_test) * 100

    # Confidence distribution — how often is model confident vs uncertain
    confident_long  = np.mean(pred_reg > 0.65) * 100
    confident_short = np.mean(pred_reg < 0.35) * 100
    uncertain       = 100 - confident_long - confident_short

    print(f"\n  Evaluation — Vol Head:")
    print(f"    MAE            : {vol_mae:.7f}")
    print(f"    RMSE           : {vol_rmse:.7f}")
    print(f"    Naive RMSE     : {naive_vol_rmse:.7f}")
    print(f"    R²             : {vol_r2:.4f}")
    print(f"    Model vs Naive : {'BETTER ✓' if vol_rmse < naive_vol_rmse else 'WORSE ✗'}")
    print(f"\n  Evaluation — Regime Head:")
    print(f"    Accuracy       : {reg_accuracy:.1f}%  (50% = random)")
    print(f"    Confident long : {confident_long:.1f}%  (regime > 0.65)")
    print(f"    Confident short: {confident_short:.1f}%  (regime < 0.35)")
    print(f"    Uncertain/flat : {uncertain:.1f}%  (0.35 – 0.65)")

    metrics = {
        "vol_mae": vol_mae, "vol_rmse": vol_rmse,
        "vol_naive_rmse": naive_vol_rmse, "vol_r2": vol_r2,
        "reg_accuracy": reg_accuracy,
        "confident_long_pct": confident_long,
        "confident_short_pct": confident_short,
        "uncertain_pct": uncertain,
    }

    for k, v in metrics.items():
        mlflow.log_metric(f"{label}_{k}", v)

    return metrics


def train_parent(ticker="^GSPC"):
    print(f"\n{'='*55}")
    print(f"  TRAINING PARENT MODEL — {ticker}")
    print(f"{'='*55}")

    df = load_parquet(ticker)
    print(f"  [+] {len(df)} rows loaded")

    train_loader, X_test, yv_test, yr_test, scaler = prepare_data(df)

    with mlflow.start_run(run_name=f"parent_{ticker.lower().replace('^','')}"):
        mlflow.log_params({
            "ticker":      ticker,
            "model_type":  "parent",
            "hidden_size": HIDDEN_SIZE,
            "num_layers":  NUM_LAYERS,
            "context_len": CONTEXT_LEN,
            "batch_size":  BATCH_SIZE,
            "epochs":      PARENT_EPOCHS,
            "lr":          LEARNING_RATE,
            "n_features":  len(FEATURES),
            "optimizer":   "Adam",
            "loss":        "MSE+0.5*BCE",
            "targets":     "realized_vol_5d + trend_regime",
        })

        model     = StockLSTM(input_size=len(FEATURES))
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        model   = train_loop(model, train_loader, optimizer, PARENT_EPOCHS, label="parent")
        metrics = evaluate(model, X_test, yv_test, yr_test, label="parent")

        out_dir     = get_output_dir(ticker)
        model_path  = os.path.join(out_dir, "parent_model.pt")
        scaler_path = os.path.join(out_dir, "parent_scaler.pkl")
        torch.save(model.state_dict(), model_path)
        joblib.dump(scaler, scaler_path)

        mlflow.log_artifact(model_path)
        mlflow.log_artifact(scaler_path)
        mlflow.set_tag("status", "completed")

    print(f"  [✓] Parent model saved\n")
    return model, feat_scaler, vol_scaler, metrics


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
        print("  [+] LSTM frozen — training heads only")
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=FINE_TUNE_LR)
        print("  [+] Fine-tuning all layers")

    df = load_parquet(ticker)
    print(f"  [+] {len(df)} rows loaded")

    train_loader, X_test, yv_test, yr_test, scaler = prepare_data(df)

    with mlflow.start_run(run_name=f"child_{ticker.lower()}_{strategy}"):
        mlflow.log_params({
            "ticker":        ticker,
            "model_type":    "child",
            "strategy":      strategy,
            "hidden_size":   HIDDEN_SIZE,
            "num_layers":    NUM_LAYERS,
            "context_len":   CONTEXT_LEN,
            "batch_size":    BATCH_SIZE,
            "epochs":        CHILD_EPOCHS,
            "lr":            FINE_TUNE_LR if strategy == "fine_tune" else LEARNING_RATE,
            "n_features":    len(FEATURES),
            "optimizer":     "Adam",
            "loss":          "MSE+0.5*BCE",
            "targets":       "realized_vol_5d + trend_regime",
            "parent_ticker": "^GSPC",
        })

        model   = train_loop(model, train_loader, optimizer, CHILD_EPOCHS, label=ticker)
        metrics = evaluate(model, X_test, yv_test, yr_test, label=ticker)

        out_dir     = get_output_dir(ticker)
        model_path  = os.path.join(out_dir, f"{ticker.lower()}_child_model.pt")
        scaler_path = os.path.join(out_dir, f"{ticker.lower()}_scaler.pkl")
        torch.save(model.state_dict(), model_path)
        joblib.dump(scaler, scaler_path)

        mlflow.log_artifact(model_path)
        mlflow.log_artifact(scaler_path)
        mlflow.set_tag("status", "completed")

    print(f"  [✓] Child model saved\n")
    return model, feat_scaler, vol_scaler, metrics


if __name__ == "__main__":
    train_parent("^GSPC")
    train_child("AAPL", strategy="fine_tune")