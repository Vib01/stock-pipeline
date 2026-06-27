"""
src/ingestion/fetch_prices.py

Block 1: live price fetcher.
Pulls the latest quote for each ticker and writes it to data/raw/prices/.
This is the ONLY job of this file -- no features, no model, nothing else.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

try:
    import yfinance as yf  # type: ignore[import]
except ImportError:  # pragma: no cover
    yf = None

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import TICKERS, FEATURES

# project_root/data/raw/prices
RAW_PRICES_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "prices"


def fetch_live_quotes(tickers: list[str]) -> dict:
    quotes = {}
    for t in tickers:
        try:
            ticker_obj = yf.Ticker(t)
            hist = ticker_obj.history(period="2d")
            if hist.empty:
                print(f"  {t}: no data returned, skipping")
                continue
            latest = hist.iloc[-1]
            prev   = hist.iloc[-2] if len(hist) > 1 else latest
            quotes[t] = {
                "price":          float(latest["Close"]),
                "previous_close": float(prev["Close"]),
                "volume":         int(latest["Volume"]),
                "day_high":       float(latest["High"]),
                "day_low":        float(latest["Low"]),
                "timestamp":      datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            print(f"  {t}: ERROR - {e}")
    return quotes


def save_quotes(quotes: dict) -> Path:
    RAW_PRICES_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RAW_PRICES_DIR / f"quotes_{stamp}.json"
    with open(out_path, "w") as f:
        json.dump(quotes, f, indent=2)
    return out_path


if __name__ == "__main__":
    quotes = fetch_live_quotes(TICKERS)
    for ticker, data in quotes.items():
        print(f"{ticker:6s}  ${data['price']}")
    out_path = save_quotes(quotes)
    print(f"\nSaved to {out_path}")