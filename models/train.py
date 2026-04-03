import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import os
import sys
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler, StandardScaler
import mlflow
from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models.lstm_model import StockLSTM, create_sequences, CONTEXT_LEN, HIDDEN_SIZE, NUM_LAYERS
from data.data_pipeline import run_pipeline, FEATURES

BATCH_SIZE    = 16
PARENT_EPOCHS = 100
CHILD_EPOCHS  = 50
LEARNING_RATE = 0.001
FINE_TUNE_LR  = 0.0005
PATIENCE      = 10
GRAD_CLIP     = 0.5
OUTPUTS_DIR   = os.path.join(os.path.dirname(__file__), "..", "outputs")
FEATURE_STORE = os.path.join(os.path.dirname(__file__), "..", "feature_store")

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("stckmind")


def load_parquet(ticker):
    path = os.path.join(FEATURE_STORE, f"{ticker.lower()}_features.parquet")
    if not os.path.exists(path):
        return run_pipeline(ticker)
    return pd.read_parquet(path)


def get_output_dir(ticker):
    path = os.path.join(OUTPUTS_DIR, ticker.lower())
    os.makedirs(path, exist_ok=True)
    return path


def prepare_data(df):
    split_row = int(len(df) * 0.8)
    train_df  = df.iloc[:split_row]

    feat_scaler = RobustScaler()
    feat_scaler.fit(train_df[FEATURES].values)

    vol_scaler = StandardScaler()
    vol_scaler.fit(train_df[["realized_vol_5d"]].values)

    df_scaled = df.copy()
    df_scaled[FEATURES] = feat_scaler.transform(df[FEATURES].values)
    df_scaled["realized_vol_5d"] = vol_scaler.transform(df[["realized_vol_5d"]].values)

    X, y_vol, y_reg = create_sequences(df_scaled)

    split = int(len(X) * 0.8)
    X_train, X_test   = X[:split],     X[split:]
    yv_train, yv_test = y_vol[:split], y_vol[split:]
    yr_train, yr_test = y_reg[:split], y_reg[split:]

    print(f"    Train sequences : {len(X_train)} | Test sequences: {len(X_test)}")
    print(f"    Vol target (scaled) — mean: {yv_train.mean():.4f}  std: {yv_train.std():.4f}")
    print(f"    Regime target       — uptrend: {yr_train.mean()*100:.1f}%")

    train_ds = TensorDataset(
        torch.tensor(X_train),
        torch.tensor(yv_train),
        torch.tensor(yr_train),
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    return train_loader, X_test, yv_test, yr_test, feat_scaler, vol_scaler


def train_loop(model, train_loader, optimizer, epochs, label=""):
    vol_loss_fn = nn.MSELoss()
    reg_loss_fn = nn.BCELoss()

    best_loss    = float("inf")
    patience_cnt = 0
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_vol_loss = 0.0
        total_reg_loss = 0.0

        for X_batch, yv_batch, yr_batch in train_loader:
            optimizer.zero_grad()
            pred_vol, pred_reg = model(X_batch)
            loss_vol = vol_loss_fn(pred_vol, yv_batch)
            loss_reg = reg_loss_fn(pred_reg, yr_batch)
            loss     = loss_vol + loss_reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            total_vol_loss += loss_vol.item()
            total_reg_loss += loss_reg.item()

        avg_vol   = total_vol_loss / len(train_loader)
        avg_reg   = total_reg_loss / len(train_loader)
        avg_total = avg_vol + avg_reg

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


def evaluate(model, X_test, yv_test, yr_test, vol_scaler, label=""):
    model.eval()
    with torch.no_grad():
        pred_vol, pred_reg = model(torch.tensor(X_test))

    pred_vol_sc = pred_vol.numpy().flatten()
    pred_reg_r  = pred_reg.numpy().flatten()
    yv_sc       = yv_test.flatten()
    yr_flat     = yr_test.flatten()

    pred_vol_orig = vol_scaler.inverse_transform(pred_vol_sc.reshape(-1,1)).flatten()
    yv_orig       = vol_scaler.inverse_transform(yv_sc.reshape(-1,1)).flatten()

    vol_mae    = np.mean(np.abs(pred_vol_orig - yv_orig))
    vol_rmse   = np.sqrt(np.mean((pred_vol_orig - yv_orig)**2))
    naive_rmse = np.sqrt(np.mean((yv_orig - yv_orig.mean())**2))
    ss_res     = np.sum((yv_orig - pred_vol_orig)**2)
    ss_tot     = np.sum((yv_orig - yv_orig.mean())**2)
    vol_r2     = 1 - ss_res / ss_tot

    reg_bin    = (pred_reg_r > 0.5).astype(int)
    reg_acc    = np.mean(reg_bin == yr_flat) * 100
    conf_long  = np.mean(pred_reg_r > 0.65) * 100
    conf_short = np.mean(pred_reg_r < 0.35) * 100
    uncertain  = 100 - conf_long - conf_short

    print(f"\n  Evaluation — Vol Head (original units):")
    print(f"    MAE            : {vol_mae:.6f}")
    print(f"    RMSE           : {vol_rmse:.6f}")
    print(f"    Naive RMSE     : {naive_rmse:.6f}")
    print(f"    R²             : {vol_r2:.4f}")
    print(f"    Model vs Naive : {'BETTER ✓' if vol_rmse < naive_rmse else 'WORSE ✗'}")
    print(f"\n  Evaluation — Regime Head:")
    print(f"    Accuracy       : {reg_acc:.1f}%  (50% = random)")
    print(f"    Confident long : {conf_long:.1f}%")
    print(f"    Confident short: {conf_short:.1f}%")
    print(f"    Uncertain/flat : {uncertain:.1f}%")

    metrics = {
        "vol_mae": vol_mae, "vol_rmse": vol_rmse,
        "vol_naive_rmse": naive_rmse, "vol_r2": vol_r2,
        "reg_accuracy": reg_acc, "conf_long_pct": conf_long,
        "conf_short_pct": conf_short, "uncertain_pct": uncertain,
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

    train_loader, X_test, yv_test, yr_test, feat_scaler, vol_scaler = prepare_data(df)

    with mlflow.start_run(run_name=f"parent_{ticker.lower().replace('^','')}"):
        mlflow.log_params({
            "ticker": ticker, "model_type": "parent",
            "hidden_size": HIDDEN_SIZE, "num_layers": NUM_LAYERS,
            "context_len": CONTEXT_LEN, "batch_size": BATCH_SIZE,
            "epochs": PARENT_EPOCHS, "lr": LEARNING_RATE,
            "n_features": len(FEATURES), "optimizer": "AdamW",
            "loss": "MSE(vol_scaled)+BCE(regime)",
            "targets": "realized_vol_5d + trend_regime",
        })

        model     = StockLSTM(input_size=len(FEATURES))
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)

        model   = train_loop(model, train_loader, optimizer, PARENT_EPOCHS, label="parent")
        metrics = evaluate(model, X_test, yv_test, yr_test, vol_scaler, label="parent")

        out_dir          = get_output_dir(ticker)
        model_path       = os.path.join(out_dir, "parent_model.pt")
        feat_scaler_path = os.path.join(out_dir, "parent_feat_scaler.pkl")
        vol_scaler_path  = os.path.join(out_dir, "parent_vol_scaler.pkl")

        torch.save(model.state_dict(), model_path)
        joblib.dump(feat_scaler, feat_scaler_path)
        joblib.dump(vol_scaler,  vol_scaler_path)

        mlflow.log_artifact(model_path)
        mlflow.log_artifact(feat_scaler_path)
        mlflow.log_artifact(vol_scaler_path)
        mlflow.set_tag("status", "completed")

    print(f"  [✓] Parent model saved\n")
    return model, feat_scaler, vol_scaler, metrics


def train_child(ticker, strategy="fine_tune"):
    print(f"\n{'='*55}")
    print(f"  TRAINING CHILD MODEL — {ticker} [{strategy}]")
    print(f"{'='*55}")

    parent_model_path = os.path.join(get_output_dir("^GSPC"), "parent_model.pt")
    if not os.path.exists(parent_model_path):
        print("[!] Parent not found — training parent first...")
        train_parent()

    model = StockLSTM(input_size=len(FEATURES))
    model.load_state_dict(torch.load(parent_model_path, weights_only=True))
    print(f"  [+] Loaded parent weights")

    if strategy == "freeze":
        for param in model.lstm.parameters():
            param.requires_grad = False
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LEARNING_RATE, weight_decay=1e-3
        )
        print("  [+] LSTM frozen — training heads only")
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=FINE_TUNE_LR, weight_decay=1e-3
        )
        print("  [+] Fine-tuning all layers")

    df = load_parquet(ticker)
    print(f"  [+] {len(df)} rows loaded")

    train_loader, X_test, yv_test, yr_test, feat_scaler, vol_scaler = prepare_data(df)

    with mlflow.start_run(run_name=f"child_{ticker.lower()}_{strategy}"):
        mlflow.log_params({
            "ticker": ticker, "model_type": "child", "strategy": strategy,
            "hidden_size": HIDDEN_SIZE, "num_layers": NUM_LAYERS,
            "context_len": CONTEXT_LEN, "batch_size": BATCH_SIZE,
            "epochs": CHILD_EPOCHS,
            "lr": FINE_TUNE_LR if strategy == "fine_tune" else LEARNING_RATE,
            "n_features": len(FEATURES), "optimizer": "AdamW",
            "loss": "MSE(vol_scaled)+BCE(regime)",
            "targets": "realized_vol_5d + trend_regime",
            "parent_ticker": "^GSPC",
        })

        model   = train_loop(model, train_loader, optimizer, CHILD_EPOCHS, label=ticker)
        metrics = evaluate(model, X_test, yv_test, yr_test, vol_scaler, label=ticker)

        out_dir          = get_output_dir(ticker)
        model_path       = os.path.join(out_dir, f"{ticker.lower()}_child_model.pt")
        feat_scaler_path = os.path.join(out_dir, f"{ticker.lower()}_feat_scaler.pkl")
        vol_scaler_path  = os.path.join(out_dir, f"{ticker.lower()}_vol_scaler.pkl")

        torch.save(model.state_dict(), model_path)
        joblib.dump(feat_scaler, feat_scaler_path)
        joblib.dump(vol_scaler,  vol_scaler_path)

        mlflow.log_artifact(model_path)
        mlflow.log_artifact(feat_scaler_path)
        mlflow.log_artifact(vol_scaler_path)
        mlflow.set_tag("status", "completed")

    print(f"  [✓] Child model saved\n")
    return model, feat_scaler, vol_scaler, metrics


if __name__ == "__main__":
    train_parent("^GSPC")
    train_child("AAPL", strategy="fine_tune")
