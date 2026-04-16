"""
walk_forward.py — Walk-forward validation for the vol direction classifier.

Methodology:
  - Fixed train window: 500 days
  - Test window: 60 days
  - Roll forward: 60 days each step
  - Retrain from scratch each fold (no leakage)
  - Log each fold to MLflow
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import mlflow
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset
from dotenv import load_dotenv
load_dotenv()

from models.lstm_model import StockLSTM, CONTEXT_LEN

MLFLOW_URI    = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
TRAIN_WINDOW  = 500
TEST_WINDOW   = 60
EPOCHS        = 40
LR            = 5e-4
BATCH_SIZE    = 16
PATIENCE      = 8
GRAD_CLIP     = 0.5
FEATURE_STORE = os.path.join(os.path.dirname(__file__), "..", "feature_store")


def get_features(df):
    exclude = {"vol_regime", "vol_direction", "price_direction",
               "forward_return_1d", "Open", "High", "Low", "Close", "Volume"}
    return [c for c in df.columns if c not in exclude]


def create_sequences(df, feature_cols):
    feat = df[feature_cols].values
    y    = df["vol_direction"].values
    X, labels = [], []
    for i in range(CONTEXT_LEN, len(feat)):
        X.append(feat[i - CONTEXT_LEN:i])
        labels.append(y[i])
    return np.array(X, dtype=np.float32), np.array(labels, dtype=np.float32).reshape(-1, 1)


def train_fold(X_train, y_train, n_features):
    y_flat = y_train.flatten()
    pos    = y_flat.sum()
    neg    = len(y_flat) - pos
    pw     = torch.tensor([neg / (pos + 1e-8)], dtype=torch.float32)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    ds        = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    loader    = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    model     = StockLSTM(input_size=n_features)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)

    best_loss, patience_cnt, best_w = float("inf"), 0, None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for Xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            total += loss.item()
        avg = total / len(loader)
        if avg < best_loss:
            best_loss, patience_cnt = avg, 0
            best_w = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                break

    if best_w:
        model.load_state_dict(best_w)
    return model


def eval_fold(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.tensor(X_test))).numpy().flatten()
    y     = y_test.flatten().astype(int)
    preds = (probs >= 0.5).astype(int)
    pers  = np.roll(y, 1); pers[0] = y[0]
    return {
        "f1":             float(f1_score(y, preds, zero_division=0)),
        "precision":      float(precision_score(y, preds, zero_division=0)),
        "recall":         float(recall_score(y, preds, zero_division=0)),
        "accuracy":       float(np.mean(preds == y)),
        "persistence_f1": float(f1_score(y, pers, zero_division=0)),
        "beats_baseline": f1_score(y, preds, zero_division=0) > f1_score(y, pers, zero_division=0),
    }


def run_walk_forward(ticker: str) -> dict:
    mlflow.set_tracking_uri(MLFLOW_URI)
    try:
        mlflow.set_experiment("stckmind_walk_forward")
    except Exception:
        pass

    path = os.path.join(FEATURE_STORE, f"{ticker.lower()}_features.parquet")
    df   = pd.read_parquet(path)
    features = get_features(df)
    df[features] = df[features].replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)

    n_total = len(df)
    folds   = []
    fold_n  = 0
    start   = 0

    print(f"\n{'='*60}")
    print(f"  WALK-FORWARD VALIDATION — {ticker}")
    print(f"  Train: {TRAIN_WINDOW}d | Test: {TEST_WINDOW}d | Roll: {TEST_WINDOW}d")
    print(f"{'='*60}")

    while start + TRAIN_WINDOW + TEST_WINDOW + CONTEXT_LEN <= n_total:
        fold_n += 1
        train_df = df.iloc[start : start + TRAIN_WINDOW].copy()
        test_df  = df.iloc[start + TRAIN_WINDOW : start + TRAIN_WINDOW + TEST_WINDOW].copy()

        scaler = RobustScaler()
        scaler.fit(train_df[features].values)

        train_sc = train_df.copy(); train_sc[features] = scaler.transform(train_df[features].values)
        test_sc  = test_df.copy();  test_sc[features]  = scaler.transform(test_df[features].values)

        X_train, y_train = create_sequences(train_sc, features)
        X_test,  y_test  = create_sequences(test_sc,  features)

        if len(X_train) < 20 or len(X_test) < 5:
            start += TEST_WINDOW
            continue

        train_start = str(train_df.index[0].date())
        test_start  = str(test_df.index[0].date())
        test_end    = str(test_df.index[-1].date())

        print(f"  Fold {fold_n:02d} | {test_start}→{test_end}", end="")

        model   = train_fold(X_train, y_train, len(features))
        metrics = eval_fold(model, X_test, y_test)
        metrics.update({"fold": fold_n, "train_start": train_start,
                        "test_start": test_start, "test_end": test_end,
                        "n_train": len(X_train), "n_test": len(X_test)})
        folds.append(metrics)

        beat = "✓" if metrics["beats_baseline"] else "✗"
        print(f" | F1={metrics['f1']:.4f} | persist={metrics['persistence_f1']:.4f} {beat}")

        try:
            with mlflow.start_run(run_name=f"wf_{ticker.lower()}_fold{fold_n:02d}"):
                mlflow.log_params({"ticker": ticker, "fold": fold_n,
                                   "train_window": TRAIN_WINDOW, "test_window": TEST_WINDOW,
                                   "test_start": test_start, "test_end": test_end})
                mlflow.log_metrics({"f1": metrics["f1"], "accuracy": metrics["accuracy"],
                                    "persistence_f1": metrics["persistence_f1"],
                                    "beats_baseline": int(metrics["beats_baseline"])})
        except Exception:
            pass

        start += TEST_WINDOW

    if not folds:
        print("  [!] Not enough data for walk-forward validation")
        return {}

    f1s      = [f["f1"] for f in folds]
    pers_f1s = [f["persistence_f1"] for f in folds]
    beats_ct = sum(f["beats_baseline"] for f in folds)

    summary = {
        "ticker": ticker, "n_folds": len(folds),
        "mean_f1": float(np.mean(f1s)), "std_f1": float(np.std(f1s)),
        "min_f1": float(np.min(f1s)),   "max_f1": float(np.max(f1s)),
        "mean_persistence": float(np.mean(pers_f1s)),
        "folds_beat_baseline": beats_ct,
        "pct_beat_baseline": beats_ct / len(folds) * 100,
    }

    try:
        with mlflow.start_run(run_name=f"wf_{ticker.lower()}_SUMMARY"):
            mlflow.log_params({"ticker": ticker, "n_folds": len(folds),
                               "train_window": TRAIN_WINDOW, "test_window": TEST_WINDOW})
            mlflow.log_metrics({"mean_f1": summary["mean_f1"], "std_f1": summary["std_f1"],
                                "mean_persistence": summary["mean_persistence"],
                                "pct_beat_baseline": summary["pct_beat_baseline"]})
    except Exception:
        pass

    print(f"\n  SUMMARY")
    print(f"  Folds            : {len(folds)}")
    print(f"  Mean F1          : {summary['mean_f1']:.4f} ± {summary['std_f1']:.4f}")
    print(f"  Mean persistence : {summary['mean_persistence']:.4f}")
    print(f"  Beat baseline    : {beats_ct}/{len(folds)} ({summary['pct_beat_baseline']:.1f}% of folds)")
    print(f"  F1 range         : [{summary['min_f1']:.4f}, {summary['max_f1']:.4f}]")

    out_path = os.path.join(os.path.dirname(__file__), "..", "outputs",
                            ticker.lower(), "walk_forward_summary.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  [saved] {out_path}\n")
    return summary


if __name__ == "__main__":
    for ticker in ["^GSPC", "AAPL"]:
        run_walk_forward(ticker)