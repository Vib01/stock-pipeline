"""
src/processing/sentiment_engineering.py

Block 6: score headlines with VADER (free, local, no API key).
Saves averaged sentiment per ticker to data/processed/sentiment/sentiment_latest.json
"""

import json
from pathlib import Path

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

PROJECT_ROOT  = Path(__file__).resolve().parents[2]
NEWS_DIR      = PROJECT_ROOT / "data" / "raw" / "news"
SENTIMENT_DIR = PROJECT_ROOT / "data" / "processed" / "sentiment"

# Replace with:
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import TICKERS, FEATURES

def load_latest_news() -> dict:
    news_files = sorted(NEWS_DIR.glob("news_*.json"))
    if not news_files:
        raise FileNotFoundError("No news files found -- run fetch_news.py first.")
    latest = news_files[-1]
    print(f"Scoring news from: {latest.name}\n")
    with open(latest) as f:
        return json.load(f)


def score_all(all_news: dict) -> tuple[dict, dict]:
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = {}
    scored_headlines = {}

    for ticker, articles in all_news.items():
        if not articles:
            sentiment_scores[ticker] = 0.0
            scored_headlines[ticker] = []
            continue

        scores = []
        headlines = []
        for article in articles:
            text = article["title"] + ". " + article["text"]
            score = analyzer.polarity_scores(text)["compound"]  # -1 to +1
            scores.append(score)
            headlines.append({
                "title": article["title"],
                "url":   article.get("url", ""),
                "score": round(score, 4),
            })
            print(f"  {ticker} | {score:+.2f} | {article['title'][:60]}")

        avg = sum(scores) / len(scores)
        sentiment_scores[ticker] = round(avg, 4)
        scored_headlines[ticker] = headlines
        print(f"  --> {ticker} avg: {avg:+.4f}\n")

    return sentiment_scores, scored_headlines


def save_sentiment(scores: dict) -> Path:
    SENTIMENT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SENTIMENT_DIR / "sentiment_latest.json"
    with open(out_path, "w") as f:
        json.dump(scores, f, indent=2)
    return out_path


def save_scored_headlines(scored_headlines: dict) -> Path:
    """Per-article VADER scores, keyed by ticker -- consumed by push_output.py
    to populate the 'headlines' block in ranking.json."""
    SENTIMENT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SENTIMENT_DIR / "scored_headlines.json"
    with open(out_path, "w") as f:
        json.dump(scored_headlines, f, indent=2)
    return out_path


if __name__ == "__main__":
    all_news = load_latest_news()
    scores, scored_headlines = score_all(all_news)

    out_path = save_sentiment(scores)
    save_scored_headlines(scored_headlines)
    print("=== SENTIMENT SCORES ===")
    for ticker, score in sorted(scores.items(), key=lambda x: -x[1]):
        bar = "█" * int((score + 1) * 10)
        print(f"{ticker:<8} {score:+.4f}  {bar}")
    print(f"\nSaved to {out_path}")