# StckMind — Volatility-Targeted Systematic Trading System

A risk-aware algorithmic trading system combining an LSTM volatility classifier with volatility-targeted position sizing. Built end-to-end with PyTorch, FastAPI, LangGraph, and MLflow.

---

<!-- SCREENSHOT: Full dashboard view — top cards + equity curve visible -->
<!-- File: screenshots/dashboard_overview.png -->
<!-- How to take: Open dashboard.html, run AAPL backtest, screenshot the full page from top to equity curve -->

---

## Architecture

```
Vol Direction Model (LSTM)
  ↓ predicts: will short-term volatility increase tomorrow?
  ↓ used for: position sizing and risk control

Position Sizing Engine
  ↓ vol_prob > 0.6  → reduce exposure (0.5–0.9×)
  ↓ vol_prob < 0.4  → increase exposure (1.1–1.5×)
  ↓ else            → hold current position
  ↓ EMA smoothing (span=5) + confidence filter

Backtest Engine
  ↓ strict chronological split (last 20% as test)
  ↓ transaction costs included (0.02% per unit change)
  ↓ no lookahead bias

Walk-Forward Validation
  ↓ 500-day train / 60-day test / rolling forward
  ↓ validates generalization across market regimes
```

**Transfer learning:** Parent model trained on S&P 500 → fine-tuned on individual tickers (AAPL, etc.)

---

## Results (AAPL, held-out test set)

| Metric | Strategy | Buy & Hold |
|--------|----------|------------|
| Total Return | +14.2% | +5.9% |
| Annualized Return | +11.5% | — |
| Sharpe Ratio | 0.54 | 0.31 |
| Sortino Ratio | 0.73 | — |
| Max Drawdown | -20.9% | -30.2% |
| Alpha vs BnH | +8.3% | — |

Vol Direction Model (S&P 500 test set): F1 = 0.645, beats persistence and majority baselines.

---

<!-- SCREENSHOT: Equity curve chart — strategy (green) vs buy-and-hold (blue dashed) -->
<!-- File: screenshots/equity_curve.png -->
<!-- How to take: Scroll to equity curve section of dashboard, screenshot just that chart panel -->

---

<!-- SCREENSHOT: Exposure vs Volatility chart — shows inverse relationship -->
<!-- File: screenshots/exposure_vs_vol.png -->
<!-- How to take: Scroll to "Exposure vs Realized Volatility" panel, screenshot it -->
<!-- This is the most important chart — it visually proves the model is working -->

---

## Stack

| Component | Technology |
|-----------|-----------|
| Model | PyTorch LSTM (hidden=48, 1 layer) |
| Training | BCEWithLogitsLoss + pos_weight for class balance |
| Transfer Learning | S&P 500 parent → ticker child |
| Features | 18 scale-invariant technical indicators |
| API | FastAPI + Redis caching |
| Agent Pipeline | LangGraph + Groq (llama-3.1-8b) |
| Experiment Tracking | MLflow |
| Frontend | Vanilla JS + Chart.js |

---

## Model Details

**Target:** `vol_direction` — binary classification of whether 5-day realized volatility will increase the next day (~50/50 class balance by construction).

**Features (18 total):**
- Returns: log_return, abs_return, ret_1/3/5, return_5d
- Volatility: vol_5/10/20, vol_ratio_5_20, vol_ratio_10_20
- Vol dynamics: vol_momentum, vol_spike, vol_compression, vol_change
- Trend: trend_strength, price_vs_ma
- Volume: volume_change

**Evaluation methodology:**
- Strict chronological train/test split (no shuffling)
- Scaler fit on train rows only
- Threshold tuned on validation set, applied to test set
- Compared against persistence baseline AND majority class baseline
- Walk-forward validation across multiple market regimes

---

<!-- SCREENSHOT: Model metrics panel — F1, precision, recall, baseline comparison -->
<!-- File: screenshots/model_metrics.png -->
<!-- How to take: Scroll to "Vol Direction Model — Classification Metrics" panel -->

---

<!-- SCREENSHOT: Walk-forward validation panel (only visible after running walk-forward) -->
<!-- File: screenshots/walk_forward.png -->
<!-- How to take: Run `python models/walk_forward.py` first, then re-run backtest on dashboard -->

---

## Project Structure

```
stockmind/
├── data/
│   └── data_pipeline.py       # Feature engineering + target construction
├── models/
│   ├── lstm_model.py           # LSTM architecture
│   ├── train.py                # Training loop, evaluation, MLflow logging
│   ├── predict.py              # Inference + position sizing
│   └── walk_forward.py         # Walk-forward cross-validation
├── backtest/
│   └── backtest.py             # Vol-scaled long backtest engine
├── agents/
│   └── analyst.py              # 4-agent LangGraph research pipeline
├── api/
│   └── main.py                 # FastAPI endpoints
├── frontend/
│   ├── index.html              # Signal analysis UI
│   └── dashboard.html          # Quant backtest dashboard
├── outputs/                    # Saved models, scalers, metrics JSON
├── feature_store/              # Cached parquet feature files
└── mlruns/                     # MLflow experiment tracking
```

---

## Setup

```bash
git clone https://github.com/yourusername/stockmind
cd stockmind
pip install -r requirements.txt
cp .env.example .env   # add GROQ_API_KEY
```

**Run the full pipeline:**

```bash
# 1. Build features
python data/data_pipeline.py

# 2. Train models (parent + child, both tickers)
python models/train.py

# 3. Run walk-forward validation
python models/walk_forward.py

# 4. Run backtest
python backtest/backtest.py

# 5. Start API
python api/main.py

# 6. View MLflow runs (separate terminal)
mlflow ui --backend-store-uri mlruns
```

Then open `frontend/dashboard.html` in your browser.

---

<!-- SCREENSHOT: MLflow UI showing training runs -->
<!-- File: screenshots/mlflow_runs.png -->
<!-- How to take: Run `mlflow ui --backend-store-uri mlruns`, open http://127.0.0.1:5000, screenshot the runs table -->

---

## Key Design Decisions

**Why volatility direction, not raw volatility?**
Predicting raw realized volatility (regression) produced RMSE worse than the naive baseline (predict mean). Binary classification of volatility direction (~50/50 balanced) is a more learnable target with meaningful F1 as evaluation metric.

**Why always long?**
Shorting during a bull market destroyed performance in early experiments (Sharpe -2.7). A long-only vol-scaling strategy is defensible as risk management, not directional alpha.

**Why transfer learning from S&P 500?**
The S&P 500 parent model learns broad market volatility patterns (2020–present). Fine-tuning on individual tickers adapts these patterns with limited per-ticker data.

**Why walk-forward validation?**
A single train/test split may be regime-dependent. Walk-forward tests whether the signal holds across multiple 60-day windows — if mean F1 beats persistence across >50% of folds, the signal is not a data artifact.

---

## Honest Limitations

- ~1,000 rows of daily data is a small sample for financial ML
- Test accuracy ~52-54% reflects the difficulty of price/vol prediction
- Backtest is on held-out data but not forward-tested in live markets
- Vol model beats persistence baseline on S&P 500; AAPL result is mixed
- Sigmoid outputs are uncalibrated — should not be interpreted as probabilities

---



*Research tool only. Not financial advice.*