import pandas as pd
import yfinance as yf
import time
import csv

df = pd.read_csv('all_tickers.csv')

# It has columns like 'Symbol', 'Name', 'Exchange', 'AssetType'

tickers = df['Symbol'].tolist()

micro_caps = []
small_caps = []

print(f"Total tickers to process: {len(tickers)}")

for i, symbol in enumerate(tickers):
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        market_cap = info.get('marketCap')

        if market_cap is None:
            print(f"{symbol}: market cap not found")
        else:
            if market_cap < 300_000_000:
                micro_caps.append([symbol, market_cap])
                print(f"{symbol}: Micro Cap (${market_cap})")
            elif market_cap < 2_000_000_000:
                small_caps.append([symbol, market_cap])
                print(f"{symbol}: Small Cap (${market_cap})")

    except Exception as e:
        print(f"{symbol}: Error fetching data - {e}")

    # Be polite with rate limits
    time.sleep(0.5)

    # Optional: Print progress every 50
    if (i+1) % 50 == 0:
        print(f"Processed {i+1}/{len(tickers)}")

# Save results to CSV
with open('micro_caps.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Symbol', 'MarketCap'])
    writer.writerows(micro_caps)

with open('small_caps.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Symbol', 'MarketCap'])
    writer.writerows(small_caps)

print(f"Saved {len(micro_caps)} micro caps and {len(small_caps)} small caps to CSV files.")