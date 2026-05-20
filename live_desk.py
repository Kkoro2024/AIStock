import os
import time
import json
import sys
from datetime import datetime, timedelta
import alpaca_trade_api as tradeapi

STOP_LOSS_PCT = 0.0300   
FINAL_TAKE_PROFIT = 0.0800

CONFIG_FILE = "alpaca_config.json"
with open(CONFIG_FILE, "r") as f:
    config = json.load(f)

api = tradeapi.REST(
    config.get("ALPACA_API_KEY", "").strip(),
    config.get("ALPACA_SECRET_KEY", "").strip(),
    config.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").replace("/v2", ""),
    api_version='v2'
)

scaled_tracking_matrix = {}

print("==============================================================")
print("🛡️ BI-DIRECTIONAL SWING RISK TRACKER MONITOR ENGAGED")
print("==============================================================")

while True:
    now = datetime.now()
    
    # 1. Skip weekends
    if now.weekday() >= 5:
        print("💤 Weekend standby active. Sleeping for 1 hour...")
        time.sleep(3600)
        continue

    # 2. Before Market Open: Calculate precise time to wait
    if now.hour < 9 or (now.hour == 9 and now.minute < 30):
        market_open_target = now.replace(hour=9, minute=30, second=0, microsecond=0)
        time_to_wait = (market_open_target - now).total_seconds()
        print(f"💤 Pre-market standby. Sleeping for {time_to_wait/60:.1f} minutes until 9:30 AM...")
        time.sleep(time_to_wait)
        continue

    # 3. After Market Close: Sleep until next day morning window
    elif now.hour >= 16:
        tomorrow_open = (now + timedelta(days=1)).replace(hour=9, minute=30, second=0, microsecond=0)
        time_to_wait = (tomorrow_open - now).total_seconds()
        print(f"💤 Post-market closed. Sleeping for {time_to_wait/3600:.1f} hours until tomorrow's open...")
        time.sleep(time_to_wait)
        continue

    # 4. Intraday Core Risk Management Loop
    else:
        try:
            positions = api.list_positions()
            for pos in positions:
                ticker = pos.symbol
                qty = float(pos.qty)  
                side = pos.side.upper() 
                entry_price = float(pos.avg_entry_price)
                current_price = float(pos.current_price)
                
                if side == "LONG":
                    return_pct = (current_price - entry_price) / entry_price
                else:
                    return_pct = (entry_price - current_price) / entry_price
                
                if ticker not in scaled_tracking_matrix:
                    scaled_tracking_matrix[ticker] = {"tier_1": False, "tier_2": False}

                if return_pct <= -STOP_LOSS_PCT:
                    print(f"🚨 [STOP LOSS] Exit limit breached on {ticker} ({side}) at {return_pct*100:.2f}%. Liquidating...")
                    api.close_position(ticker)
                    
                elif return_pct >= FINAL_TAKE_PROFIT:
                    print(f"💰 [TAKE PROFIT FINAL] Target hit on {ticker} (+{return_pct*100:.2f}%). Closing remainder...")
                    api.close_position(ticker)
                    
                elif return_pct >= 0.0300 and not scaled_tracking_matrix[ticker]["tier_1"]:
                    scale_qty = int(abs(qty) * 0.33)
                    if scale_qty >= 1:
                        print(f"💵 [TAKE PROFIT TIER 1] Scale target reached for {ticker}. Trimming 33% ({scale_qty} shares)...")
                        api.submit_order(symbol=ticker, qty=scale_qty, side='sell' if side == 'LONG' else 'buy', type='market', time_in_force='day')
                    scaled_tracking_matrix[ticker]["tier_1"] = True
                    
                elif return_pct >= 0.0550 and not scaled_tracking_matrix[ticker]["tier_2"]:
                    scale_qty = int(abs(qty) * 0.33)
                    if scale_qty >= 1:
                        print(f"💵 [TAKE PROFIT TIER 2] Scale target reached for {ticker}. Trimming additional 33% ({scale_qty} shares)...")
                        api.submit_order(symbol=ticker, qty=scale_qty, side='sell' if side == 'LONG' else 'buy', type='market', time_in_force='day')
                    scaled_tracking_matrix[ticker]["tier_2"] = True
                    
        except Exception as error:
            print(f"❌ Risk Engine Loop Error: {error}")
        
        time.sleep(30)