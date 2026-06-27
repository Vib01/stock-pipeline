"""
src/prediction/predict.py

Block 4: load the saved model, score today's live features, print the ranking.
Run AFTER fetch_prices.py and feature_engineering.py have both been run.
"""

from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR    = PROJECT_ROOT / "models"
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features" / "live_features.csv"
# Replace with:
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import TICKERS, FEATURES


def load_model() -> XGBClassifier:
    model = XGBClassifier()
    model.load_model(MODELS_DIR / "model.json")
    return model


def load_live_features() -> pd.DataFrame:
    df = pd.read_csv(FEATURES_PATH, index_col="ticker")
    return df[FEATURES]


if __name__ == "__main__":
    model = load_model()
    live = load_live_features()

    live["prob_up"] = model.predict_proba(live)[:, 1]
    ranking = live.sort_values("prob_up", ascending=False)

    print("=== TODAY'S RANKING ===")
    print(f"{'Rank':<6}{'Ticker':<8}{'P(up tomorrow)'}")
    print("-" * 28)
    for rank, (ticker, row) in enumerate(ranking.iterrows(), 1):
        bar = "█" * int(row["prob_up"] * 20)
        print(f"{rank:<6}{ticker:<8}{row['prob_up']:.4f}  {bar}")
