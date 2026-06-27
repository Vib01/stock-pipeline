"""
src/training/auto_retrain.py

Phase C: runs once daily after market close (5pm ET).
- Downloads latest trading day prices for all 20 tickers
- Appends to historical JSON files
- Retrains model on full updated history
- Logs accuracy to logs/training_log.csv
"""

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

PROJECT_ROOT   = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.config import TICKERS, FEATURES

HISTORICAL_DIR = PROJECT_ROOT / "data" / "raw" / "prices" / "historical"
MODELS_DIR     = PROJECT_ROOT / "models"
LOG_PATH       = PROJECT_ROOT / "logs" / "training_log.csv"

TRAINING_LOG_COLUMNS = [
    "timestamp", "train_rows", "test_rows", "accuracy", "baseline"
]


def update_historical(ticker: str):
    """Download latest prices and append any new rows to the historical JSON."""
    path = HISTORICAL_DIR / f"{ticker}.json"
    with open(path) as f:
        existing = json.load(f)

    last_date = existing[-1]["date"]
    print(f"  {ticker}: last date = {last_date}", end=" ")

    df = yf.download(ticker, start=last_date, auto_adjust=True, progress=False)
    if df.empty:
        print("→ no new data")
        return

    df = df.reset_index()
    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                  for c in df.columns]
    df["date"]   = df["date"].astype(str)
    df["ticker"] = ticker
    new_rows = df[["ticker", "date", "open", "high", "low", "close", "volume"]]
    new_rows = new_rows[new_rows["date"] > last_date].to_dict(orient="records")

    if not new_rows:
        print("→ already up to date")
        return

    existing.extend(new_rows)
    with open(path, "w") as f:
        json.dump(existing, f)
    print(f"→ added {len(new_rows)} new rows")


def load_and_build(ticker: str) -> pd.DataFrame:
    with open(HISTORICAL_DIR / f"{ticker}.json") as f:
        rows = json.load(f)
    df = pd.DataFrame(rows)
    df["date"]       = pd.to_datetime(df["date"])
    df               = df.sort_values("date").reset_index(drop=True)
    df["return"]     = df["close"].pct_change()
    df["ret_5"]      = df["close"].pct_change(5)
    df["ret_10"]     = df["close"].pct_change(10)
    df["ret_20"]     = df["close"].pct_change(20)
    df["volatility"] = df["return"].rolling(10).std()
    df["target"]     = (df["return"].shift(-1) > 0).astype(float)
    df.loc[df.index[-1], "target"] = np.nan
    return df


def retrain() -> dict:
    all_data = []
    for t in TICKERS:
        all_data.append(load_and_build(t))

    data      = pd.concat(all_data, ignore_index=True)
    trainable = data.dropna(subset=FEATURES + ["target"])
    cutoff    = trainable["date"].quantile(0.8)
    train     = trainable[trainable["date"] <= cutoff]
    test      = trainable[trainable["date"] > cutoff]

    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        eval_metric="logloss",
        scale_pos_weight=0.92
    )
    model.fit(train[FEATURES], train["target"])

    acc      = accuracy_score(test["target"], model.predict(test[FEATURES]))
    baseline = test["target"].mean()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(MODELS_DIR / "model.json")

    return {
        "train_rows": len(train),
        "test_rows":  len(test),
        "accuracy":   round(acc, 4),
        "baseline":   round(float(baseline), 4),
    }


def log_training(stats: dict):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRAINING_LOG_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "timestamp":  datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "train_rows": stats["train_rows"],
            "test_rows":  stats["test_rows"],
            "accuracy":   stats["accuracy"],
            "baseline":   stats["baseline"],
        })


if __name__ == "__main__":
    print(f"\n>>> Auto-retrain starting at "
          f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print("\n[1/3] Updating historical data...")
    for t in TICKERS:
        update_historical(t)

    print("\n[2/3] Retraining model...")
    stats = retrain()
    print(f"  Train rows : {stats['train_rows']}")
    print(f"  Test rows  : {stats['test_rows']}")
    print(f"  Accuracy   : {stats['accuracy']}")
    print(f"  Baseline   : {stats['baseline']}")

    print("\n[3/3] Logging...")
    log_training(stats)
    print(f"  Logged to {LOG_PATH}")

    print(f"\n>>> Done at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")