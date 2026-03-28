import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import redis
import json
import time

from models.train import train_child, train_parent
from models.predict import predict
from agents.analyst import run_analysis
from backtest.backtest import run_backtest

app = FastAPI(
    title="StckMind API",
    description="Stock prediction API powered by LSTM + Transfer Learning",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

try:
    redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)
    redis_client.ping()
    print("[+] Redis connected")
except Exception:
    redis_client = None
    print("[!] Redis not available — caching disabled")

task_store = {}

class TickerRequest(BaseModel):
    ticker: str

def cache_get(key: str):
    if redis_client:
        val = redis_client.get(key)
        return json.loads(val) if val else None
    return task_store.get(key)

def cache_set(key: str, value: dict, ttl: int = 86400):
    if redis_client:
        redis_client.setex(key, ttl, json.dumps(value))
    else:
        task_store[key] = value

def run_training_job(ticker: str):
    task_key = f"task_{ticker.lower()}"
    try:
        cache_set(task_key, {"status": "running", "started_at": time.time()}, ttl=3600)
        train_child(ticker)
        result = predict(ticker)
        cache_set(f"predict_{ticker.lower()}", result, ttl=86400)
        cache_set(task_key, {"status": "completed", "finished_at": time.time()}, ttl=3600)
    except Exception as e:
        cache_set(task_key, {"status": "failed", "error": str(e)}, ttl=3600)

@app.get("/")
def root():
    return {
        "name": "StckMind API",
        "version": "1.0.0",
        "endpoints": ["/health", "/train-parent", "/train-child", "/predict", "/analyze", "/backtest", "/status/{ticker}"]
    }

@app.get("/health")
def health():
    return {"status": "ok", "redis": "connected" if redis_client else "unavailable"}

@app.post("/train-parent")
def train_parent_endpoint(background_tasks: BackgroundTasks):
    def job():
        try:
            cache_set("task_parent", {"status": "running", "started_at": time.time()}, ttl=7200)
            train_parent("^GSPC")
            cache_set("task_parent", {"status": "completed", "finished_at": time.time()}, ttl=7200)
        except Exception as e:
            cache_set("task_parent", {"status": "failed", "error": str(e)}, ttl=7200)
    background_tasks.add_task(job)
    return {"status": "training", "detail": "Parent model training started in background"}

@app.post("/train-child")
def train_child_endpoint(request: TickerRequest, background_tasks: BackgroundTasks):
    ticker = request.ticker.strip().upper()
    background_tasks.add_task(run_training_job, ticker)
    return {"status": "training", "ticker": ticker, "check_status": f"/status/{ticker}"}

@app.post("/predict")
def predict_endpoint(request: TickerRequest):
    ticker    = request.ticker.strip().upper()
    cache_key = f"predict_{ticker.lower()}"
    cached    = cache_get(cache_key)
    if cached:
        cached["cached"] = True
        return cached
    try:
        result = predict(ticker)
        cache_set(cache_key, result, ttl=86400)
        result["cached"] = False
        return result
    except FileNotFoundError:
        raise HTTPException(404, f"No model found for {ticker}. POST /train-child first.")
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/status/{ticker}")
def get_status(ticker: str):
    status = cache_get(f"task_{ticker.lower()}")
    if not status:
        return {"ticker": ticker.upper(), "status": "no task found"}
    return {"ticker": ticker.upper(), **status}

@app.post("/backtest")
def backtest_endpoint(request: TickerRequest):
    ticker    = request.ticker.strip().upper()
    cache_key = f"backtest_{ticker.lower()}"
    cached    = cache_get(cache_key)
    if cached:
        cached["cached"] = True
        return cached
    try:
        result = run_backtest(ticker)
        result.pop("equity_curve", None)
        result.pop("bnh_curve",    None)
        result.pop("dates",        None)
        cache_set(cache_key, result, ttl=86400)
        result["cached"] = False
        return result
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/analyze")
def analyze_endpoint(request: TickerRequest):
    ticker    = request.ticker.strip().upper()
    cache_key = f"analyze_{ticker.lower()}"
    cached    = cache_get(cache_key)
    if cached:
        cached["cached"] = True
        return cached

    try:
        predictions = predict(ticker)
    except FileNotFoundError:
        raise HTTPException(404, f"No model found for {ticker}. POST /train-child first.")

    # Run backtest — attach to response
    try:
        bt = run_backtest(ticker)
        bt.pop("equity_curve", None)
        bt.pop("bnh_curve",    None)
        bt.pop("dates",        None)
    except Exception:
        bt = None

    try:
        result       = run_analysis(ticker, predictions)
        result["backtest"] = bt
        cache_set(cache_key, result, ttl=86400)
        result["cached"] = False
        return result
    except Exception as e:
        raise HTTPException(500, str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)