"""
src/training/train_model.py

One-time run: trains the XGBoost model on historical data and saves it to models/model.json.
Re-run this whenever you want to retrain on fresher data.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HISTORICAL_DIR = PROJECT_ROOT / "data" / "raw" / "prices" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"

# Replace with:
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import TICKERS, FEATURES


def load_and_build(ticker: str) -> pd.DataFrame:
    with open(HISTORICAL_DIR / f"{ticker}.json") as f:
        rows = json.load(f)
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    df["return"] = df["close"].pct_change()
    df["ret_5"] = df["close"].pct_change(5)
    df["ret_10"] = df["close"].pct_change(10)
    df["ret_20"] = df["close"].pct_change(20)
    df["volatility"] = df["return"].rolling(10).std()
    df["target"] = (df["return"].shift(-1) > 0).astype(float)
    df.loc[df.index[-1], "target"] = np.nan
    return df


if __name__ == "__main__":
    all_data = []
    for t in TICKERS:
        all_data.append(load_and_build(t))

    data = pd.concat(all_data, ignore_index=True)
    trainable = data.dropna(subset=FEATURES + ["target"])

    cutoff = trainable["date"].quantile(0.8)
    train = trainable[trainable["date"] <= cutoff]
    test = trainable[trainable["date"] > cutoff]

    X_train, y_train = train[FEATURES], train["target"]
    X_test, y_test = test[FEATURES], test["target"]

    model = XGBClassifier(
         n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        eval_metric="logloss",
        scale_pos_weight=0.92
    )
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Train rows : {len(train)}")
    print(f"Test rows  : {len(test)}")
    print(f"Accuracy   : {acc:.4f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(MODELS_DIR / "model.json")
    print(f"\nModel saved to {MODELS_DIR / 'model.json'}")