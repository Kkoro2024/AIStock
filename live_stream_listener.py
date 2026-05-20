import os
import json
import sys
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import alpaca_trade_api as tradeapi
from alpaca_trade_api.stream import Stream

# ==================================================================
# 1. CORE PARAMETERS & MODEL INITIALIZATION
# ==================================================================
CONFIG_FILE = "alpaca_config.json"
with open(CONFIG_FILE, "r") as f:
    config = json.load(f)

ALPACA_API_KEY = config.get("ALPACA_API_KEY", "").strip()
ALPACA_SECRET_KEY = config.get("ALPACA_SECRET_KEY", "").strip()
BASE_URL = config.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").strip()

# Initialize ML infrastructure in global memory once to maximize performance
scaler = joblib.load("portfolio_scaler.pkl")
model = xgb.XGBClassifier()
model.load_model("portfolio_xgb_model.json")

# Open REST handle exclusively for order routing
api_rest = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, BASE_URL, api_version='v2')

# High-speed data caching dictionaries
MINUTE_BARS_CACHE = {} 
with open("universe.json", "r") as f:
    universe_tickers = [t.replace('-', '.') for t in json.load(f)]

# ==================================================================
# 2. SELECTION ENGINE & WHIP ROUTER
# ==================================================================
def run_realtime_inference(ticker, fresh_bar):
    """Processes incoming data packets immediately upon bar closure."""
    # Manage the sliding minute cache for the active ticker
    if ticker not in MINUTE_BARS_CACHE:
        MINUTE_BARS_CACHE[ticker] = []
    
    MINUTE_BARS_CACHE[ticker].append({
        'close': fresh_bar.close,
        'high': fresh_bar.high,
        'low': fresh_bar.low,
        'open': fresh_bar.open
    })
    
    # Trim cache to keep only what is needed for feature math (last 30 minutes)
    if len(MINUTE_BARS_CACHE[ticker]) > 30:
        MINUTE_BARS_CACHE[ticker].pop(0)
        
    if len(MINUTE_BARS_CACHE[ticker]) < 26:
        return # Wait until we have enough minutes to calculate indicators cleanly

    # Convert rolling cache to a light DataFrame
    df = pd.DataFrame(MINUTE_BARS_CACHE[ticker])
    
    # High-speed micro feature calculations
    stock_ret_1m = (df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]
    stock_ret_5m = (df['close'].iloc[-1] - df['close'].iloc[-6]) / df['close'].iloc[-6]
    
    # Inline MACD histogram tracking
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd.iloc[-1] - macd_signal.iloc[-1]

    # Structure data payload directly for your XGBoost model
    feature_matrix = np.array([[
        0.0001,      # Market_Ret_1d proxy
        0.0,         # VIX_Ret_1d
        0.0,         # USD_Ret_1d
        stock_ret_1m,
        stock_ret_5m,
        stock_ret_5m, # Relative Performance proxy
        macd_hist
    ]])
    
    X_scaled = scaler.transform(feature_matrix)
    probability = model.predict_proba(X_scaled)[0, 1]
    
    # 🎯 IF SELECTION PASSES CONFIDENCE STRATUM -> TRIGER INSTANT ORDER ROUTING
    if probability >= 0.65:
        print(f"🔥 [STREAM SIGNAL] AI Breakout Detected for {ticker}! Confidence: {probability*100:.1f}%")
        try:
            account = api_rest.get_account()
            allocation = float(account.buying_power) * 0.10 # Allocate 10% of available capital per breakout
            
            # Strict enforcement of whole-share routing rules
            quantity = int(allocation // fresh_bar.close)
            
            if quantity >= 1:
                print(f" -> Streaming whole-share order: BUY {quantity} shares of {ticker}...")
                api_rest.submit_order(
                    symbol=ticker,
                    qty=quantity,
                    side='buy',
                    type='market',
                    time_in_force='day'
                )
        except Exception as order_error:
            print(f" ❌ Dynamic stream order placement failed: {order_error}")

# ==================================================================
# 3. WEBSOCKET ROUTING CALLBACKS
# ==================================================================
async def handle_incoming_bar(bar):
    """Callback listener triggered automatically when a minute bar finishes on NYSE."""
    ticker = bar.symbol
    if ticker in universe_tickers:
        run_realtime_inference(ticker, bar)

def main():
    print("==================================================================")
    print("📡 INITIALIZING ALPACO LIVE WEBSOCKET STREAM RECEIVER")
    print("==================================================================")
    print(f"Subscribing to minute aggregate channels for S&P 500 assets...")
    
    # Establish a permanent stream connection handle
    stream = Stream(
        ALPACA_API_KEY,
        ALPACA_SECRET_KEY,
        base_url=BASE_URL.replace("https://", "wss://").replace("/v2", ""),
        data_feed='iex' # Change to 'sip' if using an authenticated paid Alpaca data tier
    )
    
    # Assign the listener callback to the minute bar stream channel
    stream.subscribe_bars(handle_incoming_bar, *universe_tickers)
    
    try:
        stream.run()
    except KeyboardInterrupt:
        print("\n👋 Stream listener connection severed successfully.")
        sys.exit(0)

if __name__ == "__main__":
    main()