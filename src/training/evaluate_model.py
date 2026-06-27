"""
src/training/evaluate_model.py

Runs diagnostics on the trained model. Run this after train_model.py
to understand model behaviour before trusting the rankings.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from xgboost import XGBClassifier

PROJECT_ROOT   = Path(__file__).resolve().parents[2]
HISTORICAL_DIR = PROJECT_ROOT / "data" / "raw" / "prices" / "historical"
MODELS_DIR     = PROJECT_ROOT / "models"

TICKERS  = ["AAPL", "MSFT", "AMZN", "TSLA", "NVDA", "META", "GOOGL"]
FEATURES = ["return", "ret_5", "ret_10", "ret_20", "volatility"]


def load_and_build(ticker: str) -> pd.DataFrame:
    with open(HISTORICAL_DIR / f"{ticker}.json") as f:
        rows = json.load(f)
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["return"]     = df["close"].pct_change()
    df["ret_5"]      = df["close"].pct_change(5)
    df["ret_10"]     = df["close"].pct_change(10)
    df["ret_20"]     = df["close"].pct_change(20)
    df["volatility"] = df["return"].rolling(10).std()
    df["target"]     = (df["return"].shift(-1) > 0).astype(float)
    df.loc[df.index[-1], "target"] = np.nan
    df["ticker"] = ticker
    return df


if __name__ == "__main__":
    # Load model
    model = XGBClassifier()
    model.load_model(MODELS_DIR / "model.json")

    # Rebuild dataset
    all_data = []
    for t in TICKERS:
        all_data.append(load_and_build(t))
    data = pd.concat(all_data, ignore_index=True)
    trainable = data.dropna(subset=FEATURES + ["target"])

    cutoff = trainable["date"].quantile(0.8)
    test = trainable[trainable["date"] > cutoff]
    X_test, y_test = test[FEATURES], test["target"]

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    # --- Overall accuracy ---
    print("=" * 50)
    print("OVERALL METRICS")
    print("=" * 50)
    print(f"Test rows : {len(test)}")
    print(f"Accuracy  : {accuracy_score(y_test, preds):.4f}")
    print(f"Baseline  : {y_test.mean():.4f}  (always-predict-up)")
    print()
    print(classification_report(y_test, preds, target_names=["Down", "Up"]))

    # --- Confusion matrix ---
    cm = confusion_matrix(y_test, preds)
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(f"             Pred Down  Pred Up")
    print(f"Actual Down  {cm[0,0]:>9}  {cm[0,1]:>7}")
    print(f"Actual Up    {cm[1,0]:>9}  {cm[1,1]:>7}")
    print()

    # --- Per-ticker accuracy ---
    print("=" * 50)
    print("PER-TICKER ACCURACY (test set)")
    print("=" * 50)
    test = test.copy()
    test["pred"] = preds
    for ticker in TICKERS:
        t_data = test[test["ticker"] == ticker]
        if len(t_data) == 0:
            continue
        acc = accuracy_score(t_data["target"], t_data["pred"])
        n   = len(t_data)
        print(f"{ticker:<8} {acc:.4f}  (n={n})")
    print()

    # --- Feature importances ---
    print("=" * 50)
    print("FEATURE IMPORTANCES")
    print("=" * 50)
    importances = pd.Series(model.feature_importances_, index=FEATURES)
    for feat, imp in importances.sort_values(ascending=False).items():
        bar = "█" * int(imp * 50)
        print(f"{feat:<12} {imp:.4f}  {bar}")
    print()

    # --- Probability calibration check ---
    print("=" * 50)
    print("PROBABILITY CALIBRATION (how well prob_up predicts reality)")
    print("=" * 50)
    test["prob_up"] = probs
    test["bucket"]  = pd.cut(test["prob_up"], bins=5)
    cal = test.groupby("bucket", observed=True)["target"].mean()
    print("prob_up bucket  →  actual up rate")
    for bucket, rate in cal.items():
        print(f"  {str(bucket):<25}  {rate:.4f}")