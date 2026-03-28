# StockMind

An end-to-end stock forecasting system combining LSTM-based transfer learning with a structured LLM analysis pipeline.

## Model performance is evaluated against a naive baseline and consistently outperforms it.

---

## Overview

StckMind predicts short-term stock price movements using a two-stage pipeline:

1. **LSTM Transfer Learning** — A parent model trained on S&P 500 data captures general market dynamics. Child models are fine-tuned per ticker for improved generalization.

2. **LLM-based Analysis Pipeline** — A structured LangGraph workflow processes model outputs and recent news to generate grounded, data-driven reports.

## Architecture

```
Yahoo Finance
     |
     v
Data Pipeline (30 technical features)
     |
     v
Feature Store (Parquet)
     |
     +---> Parent Model (S&P 500, 50 epochs)
     |              |
     |              v
     +---> Child Model (fine-tuned per ticker)
                    |
                    v
            Inference Engine
            (log returns -> price range)
                    |
                    v
            FastAPI Backend
            /predict  /analyze  /train-child
                    |
          +---------+---------+
          |                   |
          v                   v
    Frontend UI        LangGraph Agents
    (HTML/CSS/JS)      Technical Analyst
                       News Summarizer
                       Report Generator
                       Critic
```

---

## Technical Details

### Model

- Architecture: 3-layer LSTM, hidden size 256, LayerNorm, dropout 0.3
- Input: 60-day context window, 30 features per day
- Output: 5-day forecast as log returns, bounded to +-5% per day via tanh
- Loss: Huber loss (robust to outliers)
- Optimizer: AdamW with weight decay
- Training: Early stopping (patience 8), gradient clipping at 1.0
- Validation: Walk-forward split — test set is always the most recent 20% of sequences

### Target Variable

The model predicts log returns rather than absolute prices:

```
log_return = log(Close_t / Close_{t-1})
price_t    = price_{t-1} * exp(log_return_t)
```

This makes predictions scale-invariant across different price regimes and time periods. Absolute price prediction fails when inference-time prices differ significantly from the training distribution.

### Features (30 total)

| Category    | Features |
|-------------|----------|
| Price       | Open, High, Low, Close, Volume |
| Trend       | EMA9, EMA21, EMA50, SMA20, SMA50, MACD, MACD signal, MACD diff, ADX |
| Momentum    | RSI14, RSI7, Stochastic K, Stochastic D, ROC10 |
| Volatility  | Bollinger upper/lower/width/pct, ATR14 |
| Volume      | OBV, VWAP |
| Derived     | High-Low pct, Close-Open pct, Price vs SMA20, Price vs SMA50 |

### Transfer Learning

Training from scratch on individual stocks is limited by data availability. The parent model learns general market structure from S&P 500 data (2022-present). Child models initialize from parent weights and fine-tune on stock-specific data with a lower learning rate (0.0001 vs 0.001).

This improves generalization on stocks with limited history and reduces training time per ticker.

### Scaling

RobustScaler (median/IQR normalization) is used instead of MinMaxScaler. This handles outliers from earnings events, market crashes, and other tail events without distorting the feature distribution. The scaler is fit on training data only and applied to the full dataset, preventing data leakage.

### Evaluation

```
Metric          Description
-------         -----------
MAE             Mean absolute error on log returns
RMSE            Root mean squared error on log returns
MAPE            Mean absolute percentage error
R2              Coefficient of determination on log returns
Naive RMSE      Baseline: predict zero return every day
```

Note: R2 near zero on return prediction is expected and consistent with the efficient market hypothesis. The relevant comparison is model RMSE vs naive RMSE — StckMind beats the flat baseline.

### Output Format

Predictions are returned as price ranges rather than point estimates:

```
Date         Daily %      Range
2026-03-30   +0.057%      [$245.51 - $252.37]
2026-03-31   +0.058%      [$244.24 - $253.94]
```

Uncertainty grows with the forecast horizon using a sqrt-of-time scaling on recent realized volatility.

---

## Agent System

Four agents run sequentially via LangGraph, powered by Groq (llama-3.1-8b-instant):

| Agent | Role |
|-------|------|
| Technical Analyst | Interprets model forecast data — direction, average move, volatility comparison |
| News Summarizer | Summarizes recent ticker news headlines factually, no interpretation added |
| Report Generator | Combines technical and news data into a structured markdown report |
| Critic | Removes invented claims, duplicate sections, and unsupported analysis |

Agents are explicitly constrained to use only provided data. The critic pass enforces this by stripping any content not grounded in the technical forecast or the headlines.

---

## Stack

| Component | Technology |
|-----------|------------|
| ML framework | PyTorch |
| Data source | yfinance |
| Technical indicators | ta-lib (ta) |
| Feature storage | Parquet |
| API | FastAPI + Uvicorn |
| Caching | Redis |
| Agent orchestration | LangGraph |
| LLM | Groq API (llama-3.1-8b-instant) |
| Frontend | HTML / CSS / JS |
| Scaling | RobustScaler (scikit-learn) |

---

## Project Structure

```
stckmind/
├── data/
│   └── data_pipeline.py       # fetch, feature engineering, parquet save
├── models/
│   ├── lstm_model.py           # model architecture, sequence builder
│   ├── train.py                # parent + child training with transfer learning
│   └── predict.py              # inference, log return reconstruction, price range
├── agents/
│   └── analyst.py              # LangGraph multi-agent pipeline
├── api/
│   └── main.py                 # FastAPI endpoints
├── frontend/
│   └── index.html              # web UI
├── feature_store/              # parquet files per ticker
├── outputs/                    # saved models and scalers per ticker
├── .env                        # GROQ_API_KEY
└── requirements.txt
```

---

## Setup

### Requirements

```
Python 3.10+
Redis (for caching)
```

### Install

```bash
git clone https://github.com/yourusername/stckmind.git
cd stckmind
pip install -r requirements.txt
```

### Environment

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_key_here
```

Get a free API key at console.groq.com.

### Run

```bash
# Step 1: fetch data and engineer features
python data/data_pipeline.py

# Step 2: train parent model (S&P 500), then child model
python models/train.py

# Step 3: start the API
python api/main.py

# Step 4: open frontend/index.html in your browser
```

### API Endpoints

```
GET  /health                   system health check
POST /train-parent             train parent model on S&P 500
POST /train-child              train child model for a ticker
POST /predict                  get 5-day price forecast
POST /analyze                  forecast + agent report
GET  /status/{ticker}          check training job status
```

Example:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL"}'
```

---

## Limitations

- Predictions are statistical estimates based on historical price patterns. They do not incorporate earnings calendars, options flow, or fundamental data.
- The model is trained on recent data (2022-present) and reflects the current market regime. Performance may degrade during structural market shifts.
- R2 near zero on return prediction is inherent to financial time series, not a model defect.
- The agent report is grounded in model output and recent headlines only. It is not a substitute for professional financial analysis.

---

## Not Financial Advice

This project is built for research and learning purposes. Nothing produced by StckMind constitutes financial advice. Do not make investment decisions based on model output.