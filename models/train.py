import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import os
import sys

from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import f1_score, precision_score, recall_score

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models.lstm_model import StockLSTM, CONTEXT_LEN, HIDDEN_SIZE

BATCH_SIZE    = 16
EPOCHS        = 60
LR            = 5e-4
PATIENCE      = 10
GRAD_CLIP     = 0.5
FEATURE_STORE = os.path.join(os.path.dirname(__file__), "..", "feature_store")
OUTPUTS_DIR   = os.path.join(os.path.dirname(__file__), "..", "outputs")


def load_parquet(ticker):
    path = os.path.join(FEATURE_STORE, f"{ticker.lower()}_features.parquet")
    return pd.read_parquet(path)


def get_features(df):
    exclude = {"vol_regime", "vol_direction", "price_direction",
               "forward_return_1d", "Open", "High", "Low", "Close", "Volume"}
    return [c for c in df.columns if c not in exclude]


def get_output_dir(ticker):
    path = os.path.join(OUTPUTS_DIR, ticker.lower())
    os.makedirs(path, exist_ok=True)
    return path


def create_sequences(df, feature_cols, target_col):
    feat = df[feature_cols].values
    y    = df[target_col].values
    X, labels = [], []
    for i in range(CONTEXT_LEN, len(feat)):
        X.append(feat[i - CONTEXT_LEN:i])
        labels.append(y[i])
    return (
        np.array(X,      dtype=np.float32),
        np.array(labels, dtype=np.float32).reshape(-1, 1),
    )


def prepare_data(df, target_col):
    df = df.copy()
    FEATURES = get_features(df)
    df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)

    split = int(len(df) * 0.8)
    scaler = RobustScaler()
    scaler.fit(df.iloc[:split][FEATURES])
    df[FEATURES] = scaler.transform(df[FEATURES])

    X, y = create_sequences(df, FEATURES, target_col)

    s1 = int(len(X) * 0.8)
    X_temp, X_test = X[:s1], X[s1:]
    y_temp, y_test = y[:s1], y[s1:]

    s2 = int(len(X_temp) * 0.8)
    X_train, X_val = X_temp[:s2], X_temp[s2:]
    y_train, y_val = y_temp[:s2], y_temp[s2:]

    print(f"    Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"    Target '{target_col}' balance — positive: {y_train.mean()*100:.1f}%")

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=BATCH_SIZE, shuffle=True
    )
    return train_loader, X_train, y_train, X_val, y_val, X_test, y_test, len(FEATURES), scaler


def baselines(y):
    y = y.flatten().astype(int)
    persistence    = np.roll(y, 1); persistence[0] = y[0]
    majority_val   = int(y.mean() >= 0.5)
    majority_pred  = np.full_like(y, majority_val)
    return {
        "persistence": f1_score(y, persistence,  zero_division=0),
        "majority":    f1_score(y, majority_pred, zero_division=0),
    }


def train_loop(model, loader, y_train):
    y_flat     = y_train.flatten()
    pos_count  = y_flat.sum()
    neg_count  = len(y_flat) - pos_count
    pos_weight = torch.tensor([neg_count / (pos_count + 1e-8)], dtype=torch.float32)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)

    best_loss, patience_cnt, best_weights = float("inf"), 0, None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for X_b, y_b in loader:
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            total += loss.item()

        avg = total / len(loader)
        print(f"    Epoch {epoch:03d}/{EPOCHS} — {avg:.5f}", end="")

        if avg < best_loss:
            best_loss, patience_cnt = avg, 0
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            print(" ✓")
        else:
            patience_cnt += 1
            print(f" (patience {patience_cnt})")
            if patience_cnt >= PATIENCE:
                print(f"    [!] Early stopping at epoch {epoch}")
                break

    if best_weights:
        model.load_state_dict(best_weights)
    return model


def find_best_threshold(probs, y):
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.1, 0.9, 81):
        preds = (probs >= t).astype(int)
        f1    = f1_score(y, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, round(float(t), 3)
    return best_t


def evaluate(model, X_val, y_val, X_test, y_test, ticker="", target=""):
    model.eval()
    with torch.no_grad():
        val_probs  = torch.sigmoid(model(torch.tensor(X_val))).numpy().flatten()
        test_probs = torch.sigmoid(model(torch.tensor(X_test))).numpy().flatten()

    y_val_int  = y_val.flatten().astype(int)
    y_test_int = y_test.flatten().astype(int)

    t     = find_best_threshold(val_probs, y_val_int)
    preds = (test_probs >= t).astype(int)

    f1   = f1_score(y_test_int,    preds, zero_division=0)
    prec = precision_score(y_test_int, preds, zero_division=0)
    rec  = recall_score(y_test_int,    preds, zero_division=0)
    acc  = float(np.mean(preds == y_test_int))
    bl   = baselines(y_test)

    beats_persistence = f1 > bl["persistence"]
    beats_majority    = f1 > bl["majority"]

    verdict = (
        "BEATS BOTH BASELINES ✓" if beats_persistence and beats_majority else
        "BEATS PERSISTENCE ONLY" if beats_persistence else
        "DOES NOT BEAT BASELINES ✗"
    )

    print(f"\n  {'='*50}")
    print(f"  EVAL — {ticker} [{target}]")
    print(f"  {'='*50}")
    print(f"  Threshold (val-tuned) : {t}")
    print(f"  Accuracy              : {acc*100:.1f}%")
    print(f"  F1  (primary)         : {f1:.4f}")
    print(f"  Precision             : {prec:.4f}")
    print(f"  Recall                : {rec:.4f}")
    print(f"  Baseline persistence  : {bl['persistence']:.4f}")
    print(f"  Baseline majority     : {bl['majority']:.4f}")
    print(f"  Beats persistence     : {'YES ✓' if beats_persistence else 'NO ✗'}")
    print(f"  Beats majority        : {'YES ✓' if beats_majority    else 'NO ✗'}")
    print(f"  Verdict               : {verdict}")

    return {
        "f1": f1, "precision": prec, "recall": rec, "accuracy": acc,
        "threshold": t,
        "persistence_f1": bl["persistence"],
        "majority_f1":    bl["majority"],
        "beats_persistence": beats_persistence,
        "beats_majority":    beats_majority,
    }


def train_model(ticker, target_col, model_name, parent_path=None):
    """
    Generic trainer for any binary classification target.
    model_name: used for saving, e.g. 'vol_model' or 'price_model'
    """
    print(f"\n  {'='*55}")
    print(f"  TRAINING {model_name.upper()} — {ticker} → {target_col}")
    print(f"  {'='*55}")

    df = load_parquet(ticker)
    train_loader, X_train, y_train, X_val, y_val, X_test, y_test, n_features, scaler = prepare_data(df, target_col)

    model = StockLSTM(input_size=n_features)

    if parent_path and os.path.exists(parent_path):
        try:
            model.load_state_dict(torch.load(parent_path, weights_only=True))
            print(f"  [+] Loaded parent weights from {parent_path}")
        except Exception as e:
            print(f"  [!] Parent weight load failed: {e} — training from scratch")

    model   = train_loop(model, train_loader, y_train)
    metrics = evaluate(model, X_val, y_val, X_test, y_test, ticker=ticker, target=target_col)

    out_dir     = get_output_dir(ticker)
    model_path  = os.path.join(out_dir, f"{model_name}.pt")
    scaler_path = os.path.join(out_dir, f"{model_name}_scaler.pkl")
    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)
    print(f"  [✓] Saved → {model_path}")

    return model, scaler, metrics, model_path


# ── Convenience wrappers ───────────────────────────────

def train_ticker(ticker, parent_path=None):
    """Train both vol and price models for a ticker."""
    print(f"\n{'#'*60}")
    print(f"  TRAINING ALL MODELS FOR {ticker}")
    print(f"{'#'*60}")

    # 1. Vol direction model (risk / position sizing)
    vol_parent = parent_path.replace("price_model", "vol_model") if parent_path else None
    _, _, vol_metrics, vol_path = train_model(
        ticker, "vol_direction", "vol_model",
        parent_path=vol_parent
    )

    # 2. Price direction model (alpha / trade direction)
    _, _, price_metrics, price_path = train_model(
        ticker, "price_direction", "price_model",
        parent_path=parent_path
    )

    return vol_path, price_path, vol_metrics, price_metrics


if __name__ == "__main__":
    # Train parent models on S&P 500
    vol_parent_path, price_parent_path, _, _ = train_ticker("^GSPC")

    # Train child models on AAPL with transfer learning
    train_ticker("AAPL", parent_path=price_parent_path)