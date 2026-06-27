# src/utils/config.py

TICKERS = [
    # Original 7
    "AAPL", "MSFT", "AMZN", "TSLA", "NVDA", "META", "GOOGL",
    # Top 13 S&P 500 additions
    "BRK-B", "JPM", "V", "JNJ", "WMT", "XOM", "UNH", "MA",
    "LLY", "AVGO", "HD", "PG", "COST",
]

FEATURES = ["return", "ret_5", "ret_10", "ret_20", "volatility"]

PRICE_WEIGHT     = 0.7
SENTIMENT_WEIGHT = 0.3