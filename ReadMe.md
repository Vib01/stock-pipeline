# Stock Prediction Pipeline

A daily stock ranking system that combines price-based ML signals with news sentiment to predict which stocks are most likely to go up the next trading day.

## What it does

Runs a full pipeline every 30 minutes during market hours:
1. Fetches live prices for 7 tickers via yfinance
2. Builds rolling price features (1/5/10/20-day returns, volatility)
3. Scores features with a trained XGBoost classifier → P(up tomorrow)
4. Fetches recent headlines via Yahoo Finance RSS
5. Scores headlines with VADER sentiment → avg sentiment per ticker
6. Blends both signals (70% price, 30% sentiment) into a final ranking

## Tickers tracked
AAPL, MSFT, AMZN, TSLA, NVDA, META, GOOGL

## Project structure

```
stock-pipeline/
├── data/
│   ├── raw/
│   │   ├── prices/
│   │   │   ├── historical/        # OHLCV history per ticker (JSON)
│   │   │   └── quotes_*.json      # Live quote snapshots
│   │   └── news/
│   │       └── news_*.json        # Raw headlines per run
│   └── processed/
│       ├── features/
│       │   └── live_features.csv  # Latest feature row per ticker
│       ├── sentiment/
│       │   └── sentiment_latest.json
│       └── final_ranking.csv      # Combined ranking output
├── models/
│   └── model.json                 # Trained XGBoost model
├── src/
│   ├── ingestion/
│   │   ├── fetch_prices.py        # Block 1: live quotes
│   │   └── fetch_news.py          # Block 3: Yahoo RSS headlines
│   ├── processing/
│   │   ├── feature_engineering.py # Block 2: rolling features
│   │   ├── sentiment_engineering.py # Block 4: VADER scoring
│   │   └── merge_features.py      # Block 5: blend + rank
│   ├── training/
│   │   ├── train_model.py         # One-time model training
│   │   └── evaluate_model.py      # Model diagnostics
│   ├── prediction/
│   │   └── predict.py             # Price-only ranking
│   └── orchestration/
│       └── pipeline.py            # Runs all blocks in order
├── .env                           # API keys (never commit this)
├── .gitignore
└── requirements.txt
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## One-time: download historical data + train model

```bash
python -c "
import yfinance as yf, json
from pathlib import Path
tickers = ['AAPL','MSFT','AMZN','TSLA','NVDA','META','GOOGL']
out = Path('data/raw/prices/historical')
out.mkdir(parents=True, exist_ok=True)
for t in tickers:
    df = yf.download(t, start='2020-01-01', auto_adjust=True).reset_index()
    df.columns = [c[0].lower() if isinstance(c,tuple) else c.lower() for c in df.columns]
    df['date'] = df['date'].astype(str)
    df['ticker'] = t
    json.dump(df[['ticker','date','open','high','low','close','volume']].to_dict(orient='records'), open(out/f'{t}.json','w'))
    print(t, 'done')
"

python src/training/train_model.py
```

## Run the pipeline

```bash
# Manual run (bypasses market-hours check)
python src/orchestration/pipeline.py --force

# Normal run (no-ops outside market hours)
python src/orchestration/pipeline.py
```

## Schedule (Windows Task Scheduler — run once to set up)

```powershell
schtasks /create /tn "StockPipeline" /tr "C:\Stockprediction\venv\Scripts\python.exe C:\Stockprediction\src\orchestration\pipeline.py" /sc minute /mo 30 /st 09:30
```

## Output

`data/processed/final_ranking.csv` — updated each run with columns:
- `prob_up` — XGBoost P(stock goes up tomorrow), 0–1
- `sentiment_norm` — VADER sentiment normalized to 0–1
- `final_score` — blended score (70% price, 30% sentiment)

## Accuracy note

Next-day direction on large-cap tech is close to a coin flip (~50-52% accuracy). The value is in the *relative ranking*, not the absolute probabilities. Track the output over time before drawing conclusions.