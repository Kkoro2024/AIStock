import pandas as pd
import json
import urllib.request

def fetch_sp500_universe():
    print("==================================================")
    print("📋 STAGE 1: FETCHING S&P 500 INVESTMENT UNIVERSE")
    print("==================================================")
    
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    
    try:
        # Create a request object and fake a real web browser header
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
        )
        
        # Open the URL and pass the HTML content straight into Pandas
        with urllib.request.urlopen(req) as response:
            html = response.read()
            
        payload = pd.read_html(html)
        df = payload[0]
        
        # Yahoo Finance requires dashes instead of dots for multi-class stocks
        raw_tickers = df['Symbol'].tolist()
        clean_tickers = [ticker.replace('.', '-') for ticker in raw_tickers]
        
        print(f"✅ Successfully retrieved {len(clean_tickers)} stocks from the S&P 500.")
        
        # Save the list to a JSON file
        with open("universe.json", "w") as f:
            json.dump(clean_tickers, f)
            
        print("Saved stock universe to 'universe.json'!")
        return clean_tickers

    except Exception as e:
        print(f"❌ Error fetching stock universe: {e}")
        return []

if __name__ == "__main__":
    fetch_sp500_universe()