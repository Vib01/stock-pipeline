"""
src/processing/merge_features.py

Block 7: blend price model probability with sentiment score into final ranking.
"""

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT   = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.config import TICKERS, FEATURES, PRICE_WEIGHT, SENTIMENT_WEIGHT



FEATURES_PATH  = PROJECT_ROOT / "data" / "processed" / "features" / "live_features.csv"
SENTIMENT_PATH = PROJECT_ROOT / "data" / "processed" / "sentiment" / "sentiment_latest.json"
MODELS_DIR     = PROJECT_ROOT / "models"


def load_price_probs() -> pd.Series:
    from xgboost import XGBClassifier
    model = XGBClassifier()
    model.load_model(MODELS_DIR / "model.json")
    live = pd.read_csv(FEATURES_PATH, index_col="ticker")
    probs = model.predict_proba(live[FEATURES])[:, 1]
    return pd.Series(probs, index=live.index, name="prob_up")


def load_sentiment() -> pd.Series:
    with open(SENTIMENT_PATH) as f:
        raw = json.load(f)
    normalized = {k: (v + 1) / 2 for k, v in raw.items()}
    return pd.Series(normalized, name="sentiment_norm")


if __name__ == "__main__":
    price_probs = load_price_probs()
    sentiment   = load_sentiment()
    

    df = pd.DataFrame({"prob_up": price_probs, "sentiment_norm": sentiment})
    df["final_score"] = (
        PRICE_WEIGHT * df["prob_up"] +
        SENTIMENT_WEIGHT * df["sentiment_norm"]
    )
    df = df.sort_values("final_score", ascending=False)

    print("=== FINAL COMBINED RANKING ===")
    print(f"(price model {int(PRICE_WEIGHT*100)}% / sentiment {int(SENTIMENT_WEIGHT*100)}%)\n")
    print(f"{'Rank':<6}{'Ticker':<8}{'Price':<10}{'Sentiment':<14}{'Final Score'}")
    print("-" * 50)
    for rank, (ticker, row) in enumerate(df.iterrows(), 1):
        bar = "█" * int(row["final_score"] * 20)
        print(
            f"{rank:<6}{ticker:<8}"
            f"{row['prob_up']:.4f}    "
            f"{row['sentiment_norm']:.4f}        "
            f"{row['final_score']:.4f}  {bar}"
        )

    out_path = PROJECT_ROOT / "data" / "processed" / "final_ranking.csv"
    from src.utils.logger import log_predictions
    log_predictions(df)
    df.to_csv(out_path)
    print(f"\nSaved to {out_path}")