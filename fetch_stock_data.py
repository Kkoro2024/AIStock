import yfinance as yf
import pandas as pd
import json
import os

def download_portfolio_data():
    print("==================================================")
    print("📥 STAGE 2: MASSIVE DATA ENGINE")
    print("==================================================")
    
    # 1. Load our S&P 500 universe
    if not os.path.exists("universe.json"):
        print("❌ Error: universe.json not found! Run get_universe.py first.")
        return
        
    with open("universe.json", "r") as f:
        tickers = json.load(f)
    
    # To keep things fast and safe for our first run, let's pull the top 100 stocks
    tickers_to_download = tickers
    print(f"Preparing to download 5 years of daily data for {len(tickers_to_download)} stocks...")

    # 2. Add macro benchmarks to correlate against the stocks
    # SPY (S&P 500 ETF), DX-Y.NYB (US Dollar), ^VIX (Market Volatility)
    macro_tickers = ['SPY', 'DX-Y.NYB', '^VIX']
    all_tickers = list(set(tickers_to_download + macro_tickers))

    # 3. Download everything in a single, parallelized batch request
    print("Downloading batch data from Yahoo Finance...")
    raw_data = yf.download(all_tickers, period="5y", interval="1d", auto_adjust=True, progress=True)
    
    if raw_data.empty:
        print("❌ Download failed. No data received.")
        return
        
    close_prices = raw_data['Close']
    
    # 4. Separate macro features from individual stocks
    print("Processing and aligning multi-stock matrix...")
    macro_df = close_prices[macro_tickers].ffill().bfill()
    stock_df = close_prices[tickers_to_download].ffill().bfill()

    # 5. Restructure into a Multi-Indexed "Long-Form" DataFrame
    # This turns columns into rows so the AI treats all stocks as parts of a whole market
    all_stock_records = []
    
    for ticker in tickers_to_download:
        if ticker in stock_df.columns:
            single_stock = pd.DataFrame({
                'Stock_Price': stock_df[ticker],
                'Market_Price': macro_df['SPY'],
                'USD_Price': macro_df['DX-Y.NYB'],
                'VIX_Price': macro_df['^VIX']
            })
            # Add a identifier column for which stock this row belongs to
            single_stock['Ticker'] = ticker
            all_stock_records.append(single_stock)
            
    # Combine everything back together into a single master spreadsheet
    master_df = pd.concat(all_stock_records)
    master_df.index.name = 'Date'
    
    # Save the raw stacked matrix
    master_df.to_csv("raw_stock_matrix.csv")
    print(f"\n✅ SUCCESS: Mass data engine complete! Saved {len(master_df)} total rows.")
    print(f"Data saved to 'raw_stock_matrix.csv'")

if __name__ == "__main__":
    download_portfolio_data()