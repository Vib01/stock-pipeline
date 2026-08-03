"""
src/deploy/push_output.py

Block 6: builds ranking.json from pipeline outputs and pushes it to a
public GitHub repo so the website widget can fetch it.

Called as a subprocess by src/orchestration/pipeline.py — no imports needed
in pipeline.py itself.

One-time setup:
  1. Create a public GitHub repo, e.g. github.com/Vib01/stock-data
  2. GitHub → Settings → Developer settings → PATs (classic)
     Scope: repo  →  copy token
  3. Add to .env (project root):
       GITHUB_TOKEN=ghp_...
       GITHUB_REPO=Vib01/stock-data
"""

import base64
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

# ── Resolve project root regardless of where this script is called from ──────
# Works whether cwd is project root (subprocess) or src/deploy/ (direct run).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO  = os.getenv("GITHUB_REPO", "Vib01/stock-data")
OUTPUT_FILE  = "ranking.json"


# ── helpers ───────────────────────────────────────────────────────────────────

def _market_status() -> str:
    from datetime import time
    import zoneinfo
    now = datetime.now(zoneinfo.ZoneInfo("America/New_York"))
    if now.weekday() >= 5:
        return "closed"
    return "open" if time(9, 30) <= now.time() <= time(16, 0) else "closed"


def _sparkline(ticker: str, days: int = 20) -> list[float]:
    try:
        df = yf.download(ticker, period=f"{days + 5}d", auto_adjust=True, progress=False)
        closes = df["Close"].dropna().tail(days).values.tolist()
        return [round(float(c), 2) for c in closes]
    except Exception:
        return []


def _live_price(ticker: str) -> tuple[float, float]:
    try:
        info  = yf.Ticker(ticker).fast_info
        price = round(float(info.last_price), 2)
        prev  = round(float(info.previous_close), 2)
        chg   = round((price - prev) / prev * 100, 2) if prev else 0.0
        return price, chg
    except Exception:
        return 0.0, 0.0


def _load_ranking() -> pd.DataFrame:
    path = PROJECT_ROOT / "data" / "processed" / "final_ranking.csv"
    if not path.exists():
        raise FileNotFoundError(f"Pipeline output not found: {path}")
    df = pd.read_csv(path, index_col=0)
    df.index.name = "ticker"
    return df.reset_index()


def _load_sentiment() -> dict:
    """Returns {ticker: float_score}. Structure is flat — no headlines inside."""
    path = PROJECT_ROOT / "data" / "processed" / "sentiment" / "sentiment_latest.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _load_headlines(ticker: str, max_headlines: int = 3) -> list[dict]:
    """
    Pull per-article VADER scores written by sentiment_engineering.py.
    Falls back to [] if that file doesn't exist yet.
    """
    path = PROJECT_ROOT / "data" / "processed" / "sentiment" / "scored_headlines.json"
    if not path.exists():
        return []

    try:
        with open(path) as f:
            raw = json.load(f)
    except Exception:
        return []

    return raw.get(ticker, [])[:max_headlines]


# ── payload builder ───────────────────────────────────────────────────────────

def build_payload() -> dict:
    ranking_df = _load_ranking()

    # Normalize column names: strip whitespace, lowercase
    # Handles "Ticker", " ticker ", "Symbol", etc. from different pipeline versions
    ranking_df.columns = ranking_df.columns.str.strip().str.lower()

    # Map common alternative column names to what we expect
    col_aliases = {"symbol": "ticker", "prob": "prob_up", "score": "final_score"}
    ranking_df = ranking_df.rename(columns=col_aliases)

    required = {"ticker", "prob_up", "sentiment_norm", "final_score"}
    missing = required - set(ranking_df.columns)
    if missing:
        raise ValueError(
            f"final_ranking.csv is missing columns: {missing}\n"
            f"Found: {ranking_df.columns.tolist()}"
        )

    records = []
    sorted_df = ranking_df.sort_values("final_score", ascending=False).reset_index(drop=True)
    for rank_idx, (_, row) in enumerate(sorted_df.iterrows(), 1):
        ticker = str(row["ticker"]).strip()
        price, change_pct = _live_price(ticker)
        # sentiment_latest.json is {ticker: score} — scores already in ranking CSV,
        # headlines are fetched live from the news cache instead
        headlines = _load_headlines(ticker)

        records.append({
            "ticker":         ticker,
            "rank":           rank_idx,
            "prob_up":        round(float(row["prob_up"]), 4),
            "sentiment_norm": round(float(row["sentiment_norm"]), 4),
            "final_score":    round(float(row["final_score"]), 4),
            "price":          price,
            "change_pct":     change_pct,
            "sparkline":      _sparkline(ticker),
            "headlines":      headlines,
        })

    return {
        "last_updated":  datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "market_status": _market_status(),
        "rankings":      records,
    }


# ── GitHub Contents API push ──────────────────────────────────────────────────

def _github_push(payload: dict) -> None:
    if not GITHUB_TOKEN:
        raise EnvironmentError("GITHUB_TOKEN not set in .env")

    content_b64 = base64.b64encode(json.dumps(payload, indent=2).encode()).decode()
    api_url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{OUTPUT_FILE}"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept":        "application/vnd.github+json",
    }

    # Fetch existing SHA so GitHub accepts the update (required for PUT on existing file)
    sha = None
    r = requests.get(api_url, headers=headers)
    if r.status_code == 200:
        sha = r.json().get("sha")

    body = {
        "message": f"chore: update ranking {payload['last_updated']}",
        "content": content_b64,
    }
    if sha:
        body["sha"] = sha

    r = requests.put(api_url, headers=headers, json=body)
    r.raise_for_status()
    print(f"[push_output] ✓ Pushed {OUTPUT_FILE} to {GITHUB_REPO}")


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    try:
        payload = build_payload()
        _github_push(payload)
    except Exception as e:
        print(f"[push_output] ✗ Failed to push: {e}")
        sys.exit(1)   # non-zero so pipeline.py logs the failure visibly


if __name__ == "__main__":
    main()