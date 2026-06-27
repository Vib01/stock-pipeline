"""
src/orchestration/pipeline.py

Block 8: runs the full pipeline in one command.
Includes a market-hours guard so it no-ops outside trading hours.
"""

import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# US Eastern offset (ET = UTC-5 standard, UTC-4 daylight)
# Simple approach: use UTC and hardcode EDT offset for now
ET_OFFSET = timedelta(hours=-4)  # EDT (summer). Change to -5 in November.
MARKET_OPEN  = 9 * 60 + 30   # 9:30 AM in minutes
MARKET_CLOSE = 16 * 60        # 4:00 PM in minutes


def is_market_open() -> bool:
    now_et = datetime.now(timezone.utc) + ET_OFFSET
    if now_et.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    minutes_since_midnight = now_et.hour * 60 + now_et.minute
    return MARKET_OPEN <= minutes_since_midnight <= MARKET_CLOSE


def run_step(script_path: str, label: str):
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=PROJECT_ROOT
    )
    if result.returncode != 0:
        print(f"ERROR: {label} failed with exit code {result.returncode}")
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run even outside market hours (for testing)"
    )
    args = parser.parse_args()

    if not args.force and not is_market_open():
        now_et = datetime.now(timezone.utc) + ET_OFFSET
        print(f"Market is closed. Current ET time: {now_et.strftime('%A %H:%M')}")
        print("Use --force to run anyway.")
        sys.exit(0)

    print(f"\n>>> Pipeline starting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    run_step("src/ingestion/fetch_prices.py",         "Block 1: Fetch live prices")
    run_step("src/processing/feature_engineering.py", "Block 2: Build features")
    run_step("src/ingestion/fetch_news.py",           "Block 3: Fetch news")
    run_step("src/processing/sentiment_engineering.py","Block 4: Score sentiment")
    run_step("src/processing/merge_features.py",      "Block 5: Blend & rank")

    print(f"\n>>> Pipeline complete at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f">>> Final ranking saved to data/processed/final_ranking.csv")