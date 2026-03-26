import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import os
import sys
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# add stockmind to root
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from models.lstm_model import StockLSTM, create_sequences, fit_scaler, scale_df, FEATURES
from data.data_pipeline import run_pipeline

# kitna clean config omg(im on cpu, increase epochs and batch size on gpu)
BATCH_SIZE = 32
PARENT_EPOCHS = 20
CHILD_EPOCHS = 10
LEARNING_RATE = 0.001
FINE_TUNE_LR = 0.0001
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
FEATURE_STORE = os.path.join(os.path.dirname(__file__), "..", "feature_store")


#helping function
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


def prepare_loaders(df, scaler):
    df_scaled = scale_df(df, scaler)
    X, y = create_sequences(df_scaled)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_ds  = TensorDataset(torch.tensor(X_test),  torch.tensor(y_test))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader, X_test, y_test


# trainin loop
def train_loop(model, train_loader, optimizer, criterion, epochs, label=""):
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"  [{label}] Epoch {epoch}/{epochs} — Loss: {avg_loss:.6f}")


#eval
def evaluate(model, X_test, y_test, scaler):
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_test)).numpy()

    # inversing scale column onli
    close_idx = FEATURES.index("Close")
    n_features = len(FEATURES)

    def inverse_close(arr):
        dummy = np.zeros((arr.shape[0] * arr.shape[1], n_features))
        dummy[:, close_idx] = arr.flatten()
        inv = scaler.inverse_transform(dummy)[:, close_idx]
        return inv.reshape(arr.shape)

    preds_inv = inverse_close(preds)
    actuals_inv = inverse_close(y_test)

    mse = np.mean((preds_inv - actuals_inv) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(preds_inv - actuals_inv))
    ss_res = np.sum((actuals_inv - preds_inv) ** 2)
    ss_tot = np.sum((actuals_inv - np.mean(actuals_inv)) ** 2)
    r2 = 1 - ss_res / ss_tot

    print(f"\n  Evaluation:")
    print(f"    MSE  : {mse:.4f}")
    print(f"    RMSE : {rmse:.4f}")
    print(f"    MAE  : {mae:.4f}")
    print(f"    R²   : {r2:.4f}")

    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


# train parent
def train_parent(ticker="^GSPC"):
    print(f"\n{'='*50}")
    print(f"  TRAINING PARENT MODEL — {ticker}")
    print(f"{'='*50}")

    df = load_parquet(ticker)
    scaler = fit_scaler(df)
    train_loader, test_loader, X_test, y_test = prepare_loaders(df, scaler)

    model = StockLSTM(input_size=len(FEATURES))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    train_loop(model, train_loader, optimizer, criterion, PARENT_EPOCHS, label="Parent")
    metrics = evaluate(model, X_test, y_test, scaler)

    # save
    out_dir = get_output_dir(ticker)
    model_path = os.path.join(out_dir, "parent_model.pt")
    scaler_path = os.path.join(out_dir, "parent_scaler.pkl")
    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)

    print(f"\n  [✓] Parent model saved → {model_path}")
    return model, scaler, metrics


#  trainig child
def train_child(ticker: str, strategy: str = "freeze"):
    print(f"\n{'='*50}")
    print(f"  TRAINING CHILD MODEL — {ticker} (strategy: {strategy})")
    print(f"{'='*50}")

    # load parent model
    parent_dir = get_output_dir("^GSPC")
    parent_model_path = os.path.join(parent_dir, "parent_model.pt")

    if not os.path.exists(parent_model_path):
        print(" Parent model not found. Training parent first...")
        train_parent()

    model = StockLSTM(input_size=len(FEATURES))
    model.load_state_dict(torch.load(parent_model_path, weights_only=True))
    print(f"  Loaded parent weights from {parent_model_path}")

    # transfer learning strategy
    if strategy == "freeze":
        for param in model.lstm.parameters():
            param.requires_grad = False
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LEARNING_RATE
        )
        print("  [+] LSTM layers frozen — training dense layers only")
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=FINE_TUNE_LR)
        print("  [+] Fine-tuning all layers with lower learning rate")

    # loading child data
    df = load_parquet(ticker)
    scaler = fit_scaler(df)
    train_loader, test_loader, X_test, y_test = prepare_loaders(df, scaler)

    criterion = nn.MSELoss()
    train_loop(model, train_loader, optimizer, criterion, CHILD_EPOCHS, label=ticker)
    metrics = evaluate(model, X_test, y_test, scaler)

   
    out_dir = get_output_dir(ticker)
    model_path  = os.path.join(out_dir, f"{ticker.lower()}_child_model.pt")
    scaler_path = os.path.join(out_dir, f"{ticker.lower()}_scaler.pkl")
    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)

    print(f"\n  Child model saved → {model_path}")
    return model, scaler, metrics



if __name__ == "__main__":
    # Step 1: train parent on S&P 500
    train_parent("^GSPC")

    # Step 2: train child on AAPL using transfer learning
    train_child("AAPL", strategy="freeze")