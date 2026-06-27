"""
src/ingestion/fetch_news.py

Block 5: fetch recent news headlines per ticker via Yahoo Finance RSS.
No API key required. Free.
Saves to data/raw/news/news_TIMESTAMP.json
"""

import json
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NEWS_DIR = PROJECT_ROOT / "data" / "raw" / "news"

# Replace with:
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import TICKERS, FEATURES


def fetch_news_for_ticker(ticker: str, max_articles: int = 10) -> list:
    url = (
        f"https://feeds.finance.yahoo.com/rss/2.0/headline"
        f"?s={ticker}&region=US&lang=en-US"
    )
    resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()

    root = ET.fromstring(resp.content)
    items = root.findall(".//item")[:max_articles]

    articles = []
    for item in items:
        title = item.findtext("title", "")
        desc  = item.findtext("description", "")[:300]
        date  = item.findtext("pubDate", "")
        link  = item.findtext("link", "")
        articles.append({
            "ticker": ticker,
            "title": title,
            "text": desc,
            "date": date,
            "url": link,
        })
    return articles


def fetch_all_news() -> dict:
    all_news = {}
    for ticker in TICKERS:
        print(f"Fetching news for {ticker}...", end=" ")
        try:
            articles = fetch_news_for_ticker(ticker)
            all_news[ticker] = articles
            print(f"{len(articles)} articles")
        except Exception as e:
            print(f"ERROR: {e}")
            all_news[ticker] = []
        time.sleep(0.5)
    return all_news


def save_news(all_news: dict) -> Path:
    NEWS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = NEWS_DIR / f"news_{stamp}.json"
    with open(out_path, "w") as f:
        json.dump(all_news, f, indent=2)
    return out_path


if __name__ == "__main__":
    news = fetch_all_news()
    out_path = save_news(news)
    print(f"\nSaved to {out_path}")
    total = sum(len(v) for v in news.values())
    print(f"Total articles: {total}")
    # Preview one headline per ticker
    print("\n--- Sample headlines ---")
    for ticker, articles in news.items():
        if articles:
            print(f"{ticker}: {articles[0]['title']}")