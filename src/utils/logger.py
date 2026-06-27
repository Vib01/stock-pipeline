"""
src/utils/logger.py

Logs every pipeline run's ranking to logs/predictions_log.csv
One row per ticker per run. actual_close filled in next day.
"""

import csv
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_PATH     = PROJECT_ROOT / "logs" / "predictions_log.csv"

COLUMNS = [
    "run_timestamp",
    "ticker",
    "price_model",
    "sentiment_norm",
    "final_score",
    "rank",
    "actual_close",
    "correct",
]


def log_predictions(ranking_df) -> Path:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not LOG_PATH.exists()
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if write_header:
            writer.writeheader()
        for rank, (ticker, row) in enumerate(ranking_df.iterrows(), 1):
            writer.writerow({
                "run_timestamp":  timestamp,
                "ticker":         ticker,
                "price_model":    round(float(row["prob_up"]), 4),
                "sentiment_norm": round(float(row["sentiment_norm"]), 4),
                "final_score":    round(float(row["final_score"]), 4),
                "rank":           rank,
                "actual_close":   "",
                "correct":        "",
            })

    return LOG_PATH