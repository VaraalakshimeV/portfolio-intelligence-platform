"""
Clear corrupted cache and update current stock prices in database
"""
import sys, os, json, shutil
from pathlib import Path
sys.path.insert(0, '.')

# Step 1: Clear corrupted cache
cache_dir = Path("data/cache")
if cache_dir.exists():
    shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    print("✅ Cache cleared")

# Step 2: Fetch live prices directly via yfinance
import yfinance as yf
from src.database.database import SessionLocal
from src.database.models import Holding

TICKERS = ['AAPL','AMZN','BA','CAT','GOOGL','GS','JNJ','JPM',
           'MSFT','NVDA','TSLA','UNH','V','WMT','XOM']

db = SessionLocal()
try:
    updated = 0
    for tkr in TICKERS:
        try:
            stock = yf.Ticker(tkr)
            hist  = stock.history(period="2d")
            if hist.empty:
                print(f"✗ No data for {tkr}")
                continue
            price = float(hist['Close'].iloc[-1])
            holdings = db.query(Holding).filter(Holding.ticker == tkr).all()
            for h in holdings:
                h.current_price = price
            print(f"✅ {tkr}: ${price:.2f}")
            updated += 1
        except Exception as e:
            print(f"✗ {tkr}: {e}")

    db.commit()
    print(f"\n🎉 Updated {updated}/{len(TICKERS)} stocks successfully!")
finally:
    db.close()