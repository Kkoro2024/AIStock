import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# ==========================================
# 1. GENERATE OFFLINE HISTORICAL MOCK DATA
# ==========================================
def generate_mock_history(ticker, date_str):
    """Generates 390 minutes (a full market day) of historical price data."""
    np.random.seed(random.randint(1, 1000)) # Randomize the day's price movement
    
    # 9:30 AM to 4:00 PM is 390 minutes
    start_time = datetime.strptime(f"{date_str} 09:30:00", "%Y-%m-%d %H:%M:%S")
    timestamps = [start_time + timedelta(minutes=i) for i in range(390)]
    
    # Simulate a starting price and minute-by-minute price swings
    base_price = random.uniform(50.0, 200.0)
    shocks = np.random.normal(0.0002, 0.0015, 390) # Volatility simulation
    price_path = base_price * np.exp(np.cumsum(shocks))
    
    df = pd.DataFrame({
        "Timestamp": timestamps,
        "Ticker": ticker,
        "Price": price_path
    })
    return df

# ==========================================
# 2. THE OFFLINE RISK SIMULATOR ENGINE
# ==========================================
def run_offline_backtest():
    # Pick a random date in the past
    past_dates = ["2023-04-12", "2024-08-19", "2025-01-15", "2025-11-04"]
    chosen_date = random.choice(past_dates)
    test_ticker = "CVNA"
    
    print("==================================================================")
    print(f"OFFLINE RISK ENGINE BACKTEST: SIMULATING HISTORICAL TRADING DAY")
    print("==================================================================")
    print(f"Target Date: {chosen_date} | Asset: {test_ticker}")
    print("Pulling historical data cache... Done. (No Alpaca Used)")
    
    # Fetch historical data block
    historical_data = generate_mock_history(test_ticker, chosen_date)
    
    # Setup our simulated execution parameters
    entry_price = historical_data.iloc[0]["Price"]
    qty = 100
    take_profit_pct = 0.05
    stop_loss_pct = 0.02
    
    print(f"Simulated Buy Order Executed at Open: {historical_data.iloc[0]['Timestamp'].strftime('%H:%M:%S')}")
    print(f"Shares Bought: {qty} | Entry Price: ${entry_price:,.2f}")
    print(f"Risk Rules Setup: Stop Loss = -{stop_loss_pct*100}% | Take Profit = +{take_profit_pct*100}%\n")
    print(f"{'TIMESTAMP':<10} | {'CURRENT PRICE':<13} | {'RETURN %':<10} | {'STATUS'}")
    print("-" * 55)
    
    trade_active = True
    
    # Loop through every minute of that historical day sequentially
    for idx, row in historical_data.iterrows():
        current_price = row["Price"]
        return_pct = (current_price - entry_price) / entry_price
        timestamp_str = row["Timestamp"].strftime("%H:%M:%S")
        
        # Print status updates every 60 minutes to avoid a wall of text
        if idx % 60 == 0:
            print(f"{timestamp_str:<10} | ${current_price:<12,.2f} | {return_pct*100:+.2f}% | Monitoring...")
            
        # Check Strategy Rules Offline
        if return_pct <= -stop_loss_pct:
            print("-" * 55)
            print(f"💥 [ALERT] RISK MANAGED: STOP LOSS HIT AT {timestamp_str}!")
            print(f"Exit Price: ${current_price:,.2f} | Final Loss: {return_pct*100:.2f}%")
            trade_active = False
            break
            
        elif return_pct >= take_profit_pct:
            print("-" * 55)
            print(f"🏆 [ALERT] TARGET REACHED: TAKE PROFIT HIT AT {timestamp_str}!")
            print(f"Exit Price: ${current_price:,.2f} | Final Profit: {return_pct*100:.2f}%")
            trade_active = False
            break
            
    if trade_active:
        final_price = historical_data.iloc[-1]["Price"]
        final_return = (final_price - entry_price) / entry_price
        print("-" * 55)
        print(f"🔔 Market Closed at 16:00:00. Position liquidated at end of day.")
        print(f"Closing Price: ${final_price:,.2f} | Total Return: {final_return*100:+.2f}%")

if __name__ == "__main__":
    run_offline_backtest()