import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame
import yfinance as yf
import json
import os
import datetime
import warnings
import time
import requests

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None 

class LiveSwingPortfolioPicker:
    def __init__(self):
        print("==========================================================")
        print("🤖 INITIALIZING BI-DIRECTIONAL QUANT PORTFOLIO ENGINE")
        print("==========================================================")
        
        try:
            self.scaler = joblib.load("portfolio_scaler.pkl")
            self.model = xgb.XGBClassifier()
            self.model.load_model("portfolio_xgb_model.json")
        except Exception as e:
            print(f"❌ [FATAL] Machine Learning model files missing: {e}")
            exit()
        
        CONFIG_FILE = "alpaca_config.json"
        with open(CONFIG_FILE, "r") as f:
            self.config = json.load(f)

        self.api = tradeapi.REST(
            self.config.get("ALPACA_API_KEY", "").strip(),
            self.config.get("ALPACA_SECRET_KEY", "").strip(),
            self.config.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").replace("/v2", ""),
            api_version='v2'
        )
        self.news_api_key = self.config.get("NEWS_API_KEY", "").strip()
        self.env_regime = os.environ.get("MARKET_REGIME", "BULL")
        
    def get_universe(self):
        try:
            with open("universe.json", "r") as f:
                raw_tickers = json.load(f)
            return [ticker.replace('-', '.') for ticker in raw_tickers]
        except Exception as e:
            print("❌ [UNIVERSE ERROR]: Failed to read universe.json")
            return []

    def fetch_daily_market_data(self, tickers):
        clean_tickers = [t for t in tickers if "." not in t and "-" not in t]
        yesterday = datetime.date.today() - datetime.timedelta(days=1)
        start_date = (yesterday - datetime.timedelta(days=150)).strftime("%Y-%m-%d")
        end_date = yesterday.strftime("%Y-%m-%d")

        chunk_size = 50  
        all_bars = []
        
        for i in range(0, len(clean_tickers), chunk_size):
            chunk = clean_tickers[i:i + chunk_size]
            try:
                bars = self.api.get_bars(symbol=chunk, timeframe=TimeFrame.Day, start=start_date, end=end_date)
                if bars and not bars.df.empty:
                    df_chunk = bars.df.reset_index()
                    all_bars.append(df_chunk)
            except Exception as e:
                print(f"❌ Error downloading batch: {e}")
            time.sleep(0.3) 
            
        return pd.concat(all_bars, ignore_index=True) if all_bars else pd.DataFrame()

    def fetch_live_macro_features(self):
        try:
            spy = yf.Ticker("SPY").history(period="5d", interval="1d")
            vix = yf.Ticker("^VIX").history(period="5d", interval="1d")
            uup = yf.Ticker("UUP").history(period="5d", interval="1d")
            
            # Extract final closing values from yesterday to match the 1-day training lag
            market_ret_1d = (spy['Close'].iloc[-1] - spy['Close'].iloc[-2]) / spy['Close'].iloc[-2]
            vix_ret_1d = (vix['Close'].iloc[-1] - vix['Close'].iloc[-2]) / vix['Close'].iloc[-2]
            usd_ret_1d = (uup['Close'].iloc[-1] - uup['Close'].iloc[-2]) / uup['Close'].iloc[-2]
            market_ret_5d_sum = spy['Close'].pct_change().iloc[-5:].sum()
            
            return market_ret_1d, vix_ret_1d, usd_ret_1d, market_ret_5d_sum
        except Exception as e:
            return 0.0001, 0.0, 0.0, 0.0005

    def check_news_risk(self, ticker):
        if not self.news_api_key:
            return False
        try:
            url = f"https://newsapi.org/v2/everything?q={ticker}&sortBy=publishedAt&pageSize=5&apiKey={self.news_api_key}"
            res = requests.get(url, timeout=5)
            if res.status_code == 200:
                articles = res.json().get("articles", [])
                blacklisted_keywords = ["lawsuit", "fraud", "bankruptcy", "sec investigation", "resign", "probe"]
                for art in articles:
                    title_text = (art.get("title") or "").lower()
                    description_text = (art.get("description") or "").lower()
                    if any(kw in title_text or kw in description_text for kw in blacklisted_keywords):
                        print(f"🚨 [NEWS RISK DETECTED] Red flags found for {ticker}. Dropping from selection queue...")
                        return True
        except Exception as e:
            print(f"⚠️ News filter error for {ticker}: {e}")
        return False

    def engineer_daily_features(self, raw_bars):
        df = raw_bars.copy()
        time_col = next((c for c in ['timestamp', 'Date', 'time'] if c in df.columns), None)
        if 'symbol' not in df.columns or time_col is None:
            return pd.DataFrame()

        df['symbol'] = df['symbol'].astype(str)
        df = df.sort_values(by=['symbol', time_col])
        df.rename(columns={'symbol': 'Ticker', 'close': 'Stock_Price'}, inplace=True)
        
        # --- ALIGNED INTERFACE: Shifting live calculations back by 1 day to match train data layout ---
        df['Stock_Ret_1d'] = df.groupby('Ticker')['Stock_Price'].transform(lambda x: x.pct_change().shift(1))
        df['Stock_Ret_5d'] = df.groupby('Ticker')['Stock_Price'].transform(lambda x: x.pct_change(periods=5).shift(1))
        
        # MACD on Lagged History
        lagged_price = df.groupby('Ticker')['Stock_Price'].shift(1)
        ema12 = lagged_price.groupby(df['Ticker']).transform(lambda x: x.ewm(span=12, adjust=False).mean())
        ema26 = lagged_price.groupby(df['Ticker']).transform(lambda x: x.ewm(span=26, adjust=False).mean())
        macd = ema12 - ema26
        
        temp_df = pd.DataFrame({'Ticker': df['Ticker'], 'macd': macd}, index=df.index)
        macd_signal = temp_df.groupby('Ticker')['macd'].transform(lambda x: x.ewm(span=9, adjust=False).mean())
        df['MACD_Hist'] = macd - macd_signal
        
        # Isolate the final rows to calculate cross-sectional ranks across our live components
        latest_market_state = pd.DataFrame(df.groupby('Ticker').tail(1).copy().reset_index(drop=True))
        
        # Calculate cross-sectional percentiles on the final live snapshot rows
        latest_market_state['Stock_Ret_1d_Rank'] = latest_market_state['Stock_Ret_1d'].rank(pct=True)
        latest_market_state['Stock_Ret_5d_Rank'] = latest_market_state['Stock_Ret_5d'].rank(pct=True)
        
        return latest_market_state

    def generate_portfolio(self):
        tickers = self.get_universe()
        if not tickers: return
            
        raw_bars = self.fetch_daily_market_data(tickers)
        if raw_bars.empty: return
            
        latest_market_state = self.engineer_daily_features(raw_bars)
        if latest_market_state.empty: return
        
        market_ret_1d, vix_ret_1d, usd_ret_1d, market_ret_5d_sum = self.fetch_live_macro_features()
        
        # Apply global macro inputs
        latest_market_state['Market_Ret_1d'] = market_ret_1d
        latest_market_state['VIX_Ret_1d'] = vix_ret_1d
        latest_market_state['USD_Ret_1d'] = usd_ret_1d
        latest_market_state['Relative_Performance_5d'] = latest_market_state['Stock_Ret_5d'] - market_ret_5d_sum
        
        # Explicit order sorting matching train_model.py feature_cols array exactly
        feature_cols = [
            'Market_Ret_1d', 'VIX_Ret_1d', 'USD_Ret_1d', 
            'Stock_Ret_1d', 'Stock_Ret_5d', 'Relative_Performance_5d', 
            'Stock_Ret_1d_Rank', 'Stock_Ret_5d_Rank', 'MACD_Hist'
        ]
        
        latest_market_state = latest_market_state.dropna(subset=feature_cols)
        if latest_market_state.empty:
            print("⚠️ Dropped all rows due to empty feature constraints.")
            return

        X_live = latest_market_state[feature_cols].values
        X_scaled = self.scaler.transform(X_live)
        
        latest_market_state['AI_Probability'] = self.model.predict_proba(X_scaled)[:, 1]
        df_ranked = latest_market_state[['Ticker', 'AI_Probability']].sort_values(by='AI_Probability', ascending=False)
        
        long_candidates = df_ranked.head(20).copy()
        long_candidates['Side'] = 'BUY'
        
        short_candidates = df_ranked.tail(20).copy()
        short_candidates['Side'] = 'SELL'
        short_candidates['AI_Probability'] = 1.0 - short_candidates['AI_Probability']
        
        combined_picks = pd.concat([long_candidates, short_candidates], ignore_index=True)
        
        # --- HYBRID RISK THRESHOLD: Drop complete 50/50 noise coin-flips below 54% ---
        combined_picks = combined_picks[combined_picks['AI_Probability'] >= 0.54]
        
        passed_records = []
        for _, row in combined_picks.iterrows():
            if not self.check_news_risk(row['Ticker']):
                passed_records.append(row)
                
        autonomous_picks = pd.DataFrame(passed_records)
        if autonomous_picks.empty:
            print("⚠️ Out of candidates: all selections dropped by confidence threshold or news risks.")
            return

        autonomous_picks['Market_Regime'] = self.env_regime
        
        autonomous_picks.to_csv("live_portfolio_selection.csv", index=False)
        print(f"✅ Selection matrix exported! Successfully validated {len(autonomous_picks)} assets via alpha-rank filters.")

if __name__ == "__main__":
    picker = LiveSwingPortfolioPicker()
    picker.generate_portfolio()