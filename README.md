# StockMind

A production-grade stock prediction and analysis system built with LSTM transfer learning, a multi-agent LLM pipeline, walk-forward backtesting, and MLflow experiment tracking. Built as a complete MLOps project — not a tutorial.

---

## What It Does

You enter a stock ticker. The system:

1. Loads a pre-trained LSTM child model for that ticker (fine-tuned from an S&P 500 parent model)
2. Predicts the next 5 trading days as log returns, converted to a price range with uncertainty bounds
3. Runs a walk-forward backtest on the test window, comparing the strategy against buy-and-hold
4. Runs 4 LLM agents sequentially to produce a data-driven analysis report
5. Returns everything to the frontend — forecast, backtest metrics, and report

---

## Architecture

```
Yahoo Finance (yfinance)
        |
        v
Data Pipeline
  - OHLCV fetch
  - 30 technical features (EMA, SMA, MACD, RSI, Bollinger, ATR, OBV, VWAP, etc.)
  - Saved as Parquet in feature_store/
        |
        +──────────────────────────────────────────────+
        |                                              |
        v                                              v
Parent Model Training                        Child Model Training
  - Ticker: ^GSPC (S&P 500)                   - Any ticker (AAPL, NVDA, etc.)
  - 100 epochs, early stopping                - Loads parent weights
  - Predicts log returns                      - Fine-tunes all layers
  - Saved to outputs/^gspc/                   - 50 epochs, early stopping
        |                                     - Saved to outputs/{ticker}/
        |                                              |
        +──────────────────────────────────────────────+
                              |
                              v
                     Inference Engine
                  - Last 30 days as context
                  - Predicts 5-day log returns
                  - Clips to +-5% per day
                  - Reconstructs prices step-by-step
                  - Adds uncertainty bands (sqrt-of-time scaling)
                              |
              +───────────────+───────────────+
              |                               |
              v                               v
     Walk-Forward Backtest           LangGraph Agent Pipeline
  - Signal: long/short/flat            Agent 1: Technical Analyst
  - 0.1% transaction cost              Agent 2: News Summarizer
  - Sharpe, Sortino, drawdown          Agent 3: Report Generator
  - Win rate, profit factor            Agent 4: Critic
  - vs buy-and-hold baseline           Powered by Groq API
              |                               |
              +───────────────+───────────────+
                              |
                              v
                        FastAPI Backend
                     Redis caching layer
                              |
                              v
                       HTML/CSS/JS Frontend
```

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| ML framework | PyTorch |
| Data source | yfinance |
| Technical indicators | ta (technical analysis library) |
| Feature storage | Parquet (pandas) |
| Scaling | RobustScaler (scikit-learn) |
| API | FastAPI + Uvicorn |
| Caching | Redis |
| Agent orchestration | LangGraph |
| LLM | Groq API (llama-3.1-8b-instant) |
| Experiment tracking | MLflow |
| Frontend | HTML / CSS / JS (single file, no framework) |

---

## Project Structure

```
stckmind/
├── data/
│   └── data_pipeline.py        # fetch OHLCV, engineer 30 features, save parquet
│
├── models/
│   ├── lstm_model.py            # LSTM architecture + sequence builder
│   ├── train.py                 # parent + child training with MLflow logging
│   └── predict.py               # inference — log returns to price range
│
├── agents/
│   └── analyst.py               # 4-agent LangGraph pipeline via Groq
│
├── backtest/
│   └── backtest.py              # walk-forward backtest, Sharpe, drawdown, alpha
│
├── api/
│   └── main.py                  # FastAPI — /predict /analyze /backtest /train-child
│
├── frontend/
│   └── index.html               # full UI — forecast + backtest + report
│
├── feature_store/               # parquet files per ticker (git-ignored)
├── outputs/                     # saved models + scalers per ticker (git-ignored)
├── mlruns/                      # MLflow experiment data (git-ignored)
├── .env                         # GROQ_API_KEY (git-ignored)
└── requirements.txt
```

---

## Setup

### Prerequisites

- Python 3.10 or higher
- Redis (for caching — optional, falls back to in-memory if unavailable)
- A Groq API key (free at console.groq.com)

### Install

```bash
git clone https://github.com/yourusername/stckmind.git
cd stckmind
pip install -r requirements.txt
```

For CPU-only PyTorch (recommended if no GPU):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Environment

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_key_here
```

Get a free key at [console.groq.com](https://console.groq.com). The free tier is sufficient.

### Redis (optional)

Redis is used for caching predictions and analysis results. If not running, the system falls back gracefully to in-memory caching.

```bash
# Fedora / RHEL
sudo dnf install redis
sudo systemctl start redis
sudo systemctl enable redis

# Ubuntu / Debian
sudo apt install redis-server
sudo systemctl start redis
```

---

## Running the System

### Step 1 — Fetch data and engineer features

```bash
python data/data_pipeline.py
```

This fetches OHLCV data for `^GSPC` and `AAPL` from Yahoo Finance (2022 to present), calculates 30 technical indicators, and saves parquet files to `feature_store/`.

To add a new ticker:

```python
from data.data_pipeline import run_pipeline
run_pipeline("NVDA")
```

### Step 2 — Train models

```bash
python models/train.py
```

This trains:
- Parent model on S&P 500 (`^GSPC`) — 100 epochs with early stopping
- Child model on AAPL — fine-tuned from parent weights, 50 epochs

All runs are logged to MLflow automatically.

To train a child model for a different ticker:

```python
from models.train import train_child
train_child("NVDA", strategy="fine_tune")
```

Training time on CPU: approximately 5-10 minutes per model.

### Step 3 — Start the API

```bash
python api/main.py
```

The API starts at `http://localhost:8000`. Auto-documentation is available at `http://localhost:8000/docs`.

### Step 4 — Open the frontend

Open `frontend/index.html` directly in your browser. No build step required.

### Step 5 — View MLflow experiments (optional)

```bash
mlflow ui --backend-store-uri mlruns
```

Open `http://localhost:5000`. Click the "stckmind" experiment to see all training runs with parameters, metrics, and loss curves.

---

## API Reference

All POST endpoints accept `{"ticker": "AAPL"}` as the request body.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info and endpoint list |
| `/health` | GET | Health check, Redis status |
| `/train-parent` | POST | Train parent model on S&P 500 (background) |
| `/train-child` | POST | Train child model for a ticker (background) |
| `/predict` | POST | 5-day forecast with price ranges |
| `/analyze` | POST | Forecast + backtest + agent report |
| `/backtest` | POST | Walk-forward backtest metrics only |
| `/status/{ticker}` | GET | Check background training job status |

### Example — Get prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL"}'
```

Response:

```json
{
  "ticker": "AAPL",
  "last_known_date": "2026-03-27",
  "last_known_close": 248.80,
  "daily_volatility": 1.377,
  "forecast": [
    {
      "date": "2026-03-30",
      "predicted_close": 250.20,
      "low_estimate": 246.76,
      "high_estimate": 253.65,
      "pct_change": 0.563
    }
  ]
}
```

### Example — Full analysis

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL"}'
```

Returns forecast + backtest metrics + 4-agent report in one response.

### Example — Train a new ticker

```bash
# Start training in background
curl -X POST http://localhost:8000/train-child \
  -H "Content-Type: application/json" \
  -d '{"ticker": "NVDA"}'

# Check status
curl http://localhost:8000/status/NVDA
```

---

## Model Design

### Why Transfer Learning

Training a fresh LSTM from scratch per ticker is limited by data availability. Individual stock data from 2022-present gives roughly 800 rows — not much for a sequence model.

The solution: train one parent model on S&P 500 index data, which represents the general behavior of the US equity market. Child models for individual stocks initialize from these parent weights and fine-tune on stock-specific data. The parent provides a starting point that already understands market dynamics — momentum, mean reversion, volatility clustering — which generalizes across tickers.

### Architecture

```
Input: (batch, 30 context_days, 30 features)
  -> LSTM (hidden=64, layers=1)
  -> LayerNorm
  -> Linear(64 -> 32)
  -> ReLU
  -> Linear(32 -> 5)
Output: (batch, 5)  — 5 predicted log returns
```

Kept deliberately simple. A 1-layer LSTM with 64 hidden units is appropriate for ~800 training sequences. Larger models overfit immediately on this dataset size.

### Why Log Returns, Not Prices

Predicting absolute prices fails at inference time because the model's training distribution (prices from 2022-2024) differs from the current price level. A model trained when AAPL was at $150 cannot reliably predict when it is at $250.

Log returns are scale-invariant:

```
log_return = log(Close_t / Close_{t-1})
```

The model predicts percentage movements, which are consistent across all price levels and time periods. Prices are reconstructed step-by-step at inference:

```
price_t = price_{t-1} * exp(log_return_t)
```

### Features (30 total)

```
Price:      Open, High, Low, Close, Volume
Trend:      EMA9, EMA21, EMA50, SMA20, SMA50, MACD, MACD signal,
            MACD diff, ADX
Momentum:   RSI14, RSI7, Stochastic K, Stochastic D, ROC10
Volatility: Bollinger upper/lower/width/pct, ATR14
Volume:     OBV, VWAP
Derived:    High-Low%, Close-Open%, Price vs SMA20, Price vs SMA50
```

### Scaling

RobustScaler (median/IQR normalization) is used over MinMaxScaler. Stock data contains frequent outliers from earnings events, gap opens, and market dislocations. MinMaxScaler distorts the entire feature distribution when a single outlier is present. RobustScaler is resistant to these events.

The scaler is fit on training data only (first 80% of rows) to prevent data leakage, then applied to the full dataset.

### Walk-Forward Validation

Train and test splits are strictly temporal — the test set is always the most recent 20% of sequences. This mirrors real deployment conditions where a model trained on past data is evaluated on future data it has never seen.

Random shuffling of time series data is incorrect — it leaks future information into training and produces optimistically biased evaluation metrics.

---

## Backtesting

The backtest module simulates a trading strategy on the test window using model predictions.

### Strategy

For each day in the test window:
- Use the previous 30 days as context
- If predicted return > threshold (0.03%): go long
- If predicted return < -threshold: go short
- Otherwise: stay flat
- Apply 0.1% transaction cost on position changes

### Metrics

| Metric | Description |
|--------|-------------|
| Total return | Portfolio return over the test period |
| Annualized return | Return scaled to one year |
| Sharpe ratio | Risk-adjusted return (annualized, risk-free = 0) |
| Sortino ratio | Downside risk-adjusted return |
| Max drawdown | Largest peak-to-trough decline |
| Win rate | Percentage of trades that were profitable |
| Profit factor | Gross profit divided by gross loss |
| Alpha vs BnH | Return above buy-and-hold baseline |

### Honest Assessment

Direction accuracy on the test set is approximately 50-52%, marginally above random. This is expected and consistent with the Efficient Market Hypothesis — publicly available price and technical data alone does not reliably predict short-term returns.

The backtest is included to demonstrate the complete MLOps workflow: train, evaluate, simulate, measure. A real production system would require additional signal sources (earnings calendars, options flow, alternative data, fundamental data) to generate consistent alpha.

---

## Agent System

Four agents run sequentially via LangGraph. Each agent receives only the data it is given — no hallucinated macroeconomic context.

```
Technical Analyst
  Input:  5-day forecast table, 30-day realized volatility
  Output: trend direction, average daily move, volatility assessment
      |
      v
News Summarizer
  Input:  5 most recent Yahoo Finance headlines for the ticker
  Output: factual 2-sentence summary of headlines only
      |
      v
Report Generator
  Input:  technical analysis + news summary + forecast table
  Output: structured markdown report with stance (BULLISH/BEARISH/NEUTRAL)
      |
      v
Critic
  Input:  draft report
  Output: cleaned report — removes invented claims, duplicate sections,
          unsupported analysis
```

The Critic pass enforces grounding. Any content not derivable from the provided forecast data or headlines is stripped.

---

## MLflow Experiment Tracking

Every training run logs:

- All hyperparameters (hidden size, learning rate, context length, batch size, optimizer, loss function, number of features)
- Per-epoch training loss
- Evaluation metrics (MAE, RMSE, R², direction accuracy, prediction std)
- Model artifact (.pt file)
- Scaler artifact (.pkl file)
- Run metadata (duration, source file, git commit hash)

To view:

```bash
mlflow ui --backend-store-uri mlruns
# open http://localhost:5000
# select "stckmind" experiment from the left panel
```

---

## Limitations

These are real limitations, not disclaimers.

**Model accuracy**: Direction accuracy is ~50-52% on the test set. Price/technical features alone do not provide a reliable edge on short-term equity returns. This is a known property of efficient markets, not a failure of implementation.

**Data volume**: Four years of daily data (~800 rows per ticker) is a small dataset for a sequence model. The transfer learning approach partially mitigates this but does not eliminate it.

**Single asset class**: The system only handles US equities via Yahoo Finance. Crypto, FX, and futures are not tested.

**No fundamental data**: Earnings, revenue, P/E ratios, and other fundamental signals are not included. These are standard inputs in production quant systems.

**Backtest assumptions**: The backtest assumes perfect execution at close prices with a flat 0.1% transaction cost. Real-world slippage, market impact, and variable spread are not modeled.

---

## Extending the System

### Add a new ticker

```bash
# From project root
python -c "
from data.data_pipeline import run_pipeline
from models.train import train_child
run_pipeline('TSLA')
train_child('TSLA', strategy='fine_tune')
"
```

Then request analysis via the API or frontend.

### Add more features

Edit `FEATURES` list in `data/data_pipeline.py`. Add any indicator supported by the `ta` library. The model input size adjusts automatically from `len(FEATURES)`.

### Swap the LLM

In `agents/analyst.py`, replace the Groq model:

```python
llm = ChatGroq(model="llama-3.3-70b-versatile", ...)
```

Any LangChain-compatible LLM works as a drop-in replacement.

### Connect to DagsHub for remote MLflow

```python
import dagshub
dagshub.init(repo_owner="yourusername", repo_name="stckmind", mlflow=True)
```

Add your DagsHub credentials to `.env` and set `MLFLOW_TRACKING_URI`.

---

## Disclaimer

This project is built for research and learning. Nothing produced by StockMind constitutes financial advice. Do not make investment decisions based on model output.