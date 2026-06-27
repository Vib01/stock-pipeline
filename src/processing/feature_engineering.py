"""
src/processing/feature_engineering.py

Block 2: live feature builder.
Takes historical price data + the latest live quote, and produces ONE row
per ticker with the same 5 features the model was trained on.

Run this AFTER fetch_prices.py has saved at least one quotes_*.json file.
"""

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HISTORICAL_DIR = PROJECT_ROOT / "data" / "raw" / "prices" / "historical"
LIVE_QUOTES_DIR = PROJECT_ROOT / "data" / "raw" / "prices"

# Replace with:
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import TICKERS, FEATURES

def load_historical(ticker: str) -> pd.DataFrame:
    """Load the historical OHLCV JSON you copied over from the MVP project."""
    path = HISTORICAL_DIR / f"{ticker}.json"
    with open(path) as f:
        rows = json.load(f)
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_latest_quote_file() -> dict:
    """Find the most recently saved quotes_*.json file and load it."""
    quote_files = sorted(LIVE_QUOTES_DIR.glob("quotes_*.json"))
    if not quote_files:
        raise FileNotFoundError(
            "No quotes_*.json found -- run fetch_prices.py first."
        )
    latest_file = quote_files[-1]
    print(f"Using live quote file: {latest_file.name}")
    with open(latest_file) as f:
        return json.load(f)


def build_live_features(ticker: str, live_quotes: dict) -> pd.Series:
    """
    Append today's live price as a new row onto the historical series,
    recompute the rolling features, and return ONLY the new (live) row's
    features -- this is what gets fed to the model.
    """
    hist = load_historical(ticker)

    live_price = live_quotes[ticker]["price"]
    today_row = pd.DataFrame(
        [{"date": pd.Timestamp.now().normalize(), "close": live_price}]
    )

    combined = pd.concat([hist[["date", "close"]], today_row], ignore_index=True)
    combined = combined.sort_values("date").reset_index(drop=True)

    combined["return"] = combined["close"].pct_change()
    combined["ret_5"] = combined["close"].pct_change(5)
    combined["ret_10"] = combined["close"].pct_change(10)
    combined["ret_20"] = combined["close"].pct_change(20)
    combined["volatility"] = combined["return"].rolling(10).std()

    # The last row is the one we just appended -- today's live features
    live_row = combined.iloc[-1]
    return live_row[FEATURES]


if __name__ == "__main__":
    live_quotes = load_latest_quote_file()

    all_live_features = {}
    for ticker in TICKERS:
        feats = build_live_features(ticker, live_quotes)
        all_live_features[ticker] = feats
        print(f"\n{ticker}:")
        print(feats)

    # Save as a single DataFrame for Block 3 to consume
    live_df = pd.DataFrame(all_live_features).T
    live_df.index.name = "ticker"
    out_path = PROJECT_ROOT / "data" / "processed" / "features" / "live_features.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    live_df.to_csv(out_path)
    print(f"\nSaved live features to {out_path}")

