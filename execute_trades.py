import pandas as pd
import alpaca_trade_api as tradeapi
import os
import json
import datetime
import subprocess

# ==========================================
# CENTRALIZED CREDENTIAL MANAGEMENT
# ==========================================
CONFIG_FILE = "alpaca_config.json"
if not os.path.exists(CONFIG_FILE):
    raise FileNotFoundError(f"[CRITICAL] Master configuration file '{CONFIG_FILE}' missing from directory!")

with open(CONFIG_FILE, "r") as f:
    config = json.load(f)

ALPACA_API_KEY = config.get("ALPACA_API_KEY", "").strip()
ALPACA_SECRET_KEY = config.get("ALPACA_SECRET_KEY", "").strip()
ALPACA_BASE_URL = config.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").strip()

if ALPACA_BASE_URL.endswith("/v2"):
    ALPACA_BASE_URL = ALPACA_BASE_URL.replace("/v2", "")

# ==========================================
# AUTOMATED LEDGER & GITHUB DASHBOARD SYNC
# ==========================================
def log_trade_to_csv(ticker, side, qty, est_price, total_cost, probability, regime):
    """Appends order routing metadata into a permanent CSV ledger for daily tracking."""
    log_file = "trade_execution_log.csv"
    
    log_data = {
        "Timestamp": [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "Ticker": [ticker],
        "Side": [side.upper()],
        "Quantity": [qty],
        "Est_Entry_Price": [round(est_price, 2)],
        "Total_Capital_Allocated": [round(total_cost, 2)],
        "AI_Confidence": [round(probability, 2)],
        "Market_Regime": [regime]
    }
    df_new = pd.DataFrame(log_data)
    
    # Append to file if it already exists, write fresh with headers if it doesn't
    if os.path.exists(log_file):
        df_new.to_csv(log_file, mode='a', header=False, index=False)
    else:
        df_new.to_csv(log_file, mode='w', header=True, index=False)

def push_updates_to_github():
    """Automatically commits and pushes the updated ledger and index.html to GitHub Pages."""
    print("\n🔄 Syncing trade ledger and dashboard to GitHub Pages...")
    try:
        # Stage the necessary website files
        subprocess.run(["git", "add", "trade_execution_log.csv", "index.html"], check=True)
        
        # Commit changes with a clean, automated timestamp
        commit_message = f"Automated ledger sync: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        subprocess.run(["git", "commit", "-m", commit_message], check=True)
        
        # Push live to your main branch
        subprocess.run(["git", "push", "origin", "main"], check=True)
        print("✅ SUCCESS: GitHub Pages dashboard updated successfully!")
        
    except Exception as e:
        print(f"⚠️ Warning: Auto-push to GitHub failed. Error: {e}")

# ==========================================
# CORE ORDER EXECUTION ENGINE
# ==========================================
def run_order_execution():
    print("==================================================")
    print("💼 ALPACA UNLIMITED BROKER EXECUTION ENGINE")
    print("==================================================")
    
    try:
        api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, api_version='v2')
        account = api.get_account()
    except Exception as e:
        print(f"❌ Connection Failed: Check API keys in {CONFIG_FILE}. Error: {e}")
        return

    if account.trading_blocked:
        print("❌ Account Error: Your Alpaca trading account is blocked!")
        return

    # 🟢 CAPITAL RISK FIXED: Swapped buying_power for liquid cash to align with overnight swing limits
    available_capital = float(account.cash)
    print(f"Connected to Alpaca Environment via unified configuration.")
    print(f"Account Equity: ${float(account.portfolio_value):,.2f} | Liquid Cash Balance: ${available_capital:,.2f}")
    print("-" * 50)

    sheet_file = 'live_portfolio_selection.csv'
    if not os.path.exists(sheet_file):
        print(f"❌ Error: {sheet_file} not found! Run live_portfolio.py first.")
        return
        
    df_picks = pd.read_csv(sheet_file)
    if df_picks.empty:
        print("⚠️ Warning: Selection sheet is empty. No trades to execute.")
        return

    total_found = len(df_picks)
    print(f"[FILTER DISABLED] Deploying diversified capital across ALL {total_found} qualified assets.")

    # --- 1. EVALUATE MACRO SWITCH FROM SELECTION PAYLOAD ---
    market_regime = "BULL" 
    if 'Market_Regime' in df_picks.columns:
        market_regime = df_picks['Market_Regime'].iloc[0]
        
    print(f"[ENGINE REGIME STATUS] Detected Market Trend Framework: {market_regime}")
    
    if market_regime == "BEAR":
        print("🚨 BEAR MARKET REGIME ACTIVE: Scaling down capital deployment metrics by 75%.")
        risk_scalar = 0.25
    else:
        print("🟢 BULL MARKET REGIME ACTIVE: Full allocation parameters authorized.")
        risk_scalar = 0.95

    # --- 2. RISK ALLOCATION CAPACITY CALCULATION ---
    cash_pool = available_capital * risk_scalar
    allocation_per_stock = cash_pool / total_found
    print(f"Allocating a precise boundary of ${allocation_per_stock:,.2f} per position.\n")

    # --- 3. SUBMIT REAL ORDERS ---
    for index, row in df_picks.iterrows():
        ticker = row['Ticker']
        
        prob_cols = ['AI_Breakout_Probability', 'Probability', 'probability']
        probability = 0.0
        for col in prob_cols:
            if col in row:
                probability = row[col]
                break
        if probability < 1.0 and probability > 0:
            probability *= 100
            
        # Hardcoding the side to 'buy' for your long-only swing portfolio engine
        trade_side = 'buy'
            
        print(f"Processing Rank {index+1}: {ticker} ({trade_side.upper()} | AI Confidence: {probability:.1f}%)")
        
        try:
            latest_trade = api.get_latest_trade(ticker)
            current_price = float(latest_trade.price)
            
            # ENFORCING STRICT WHOLE SHARES
            quantity = int(allocation_per_stock // current_price)
            
            if quantity >= 1:
                total_cost = quantity * current_price
                print(f" -> Routing Order: {trade_side.upper()} {quantity} whole shares of {ticker} (@ ~${current_price:.2f})...")
                
                # TRANSMIT ROUTE TO ALPACA SERVER
                api.submit_order(
                    symbol=ticker,
                    qty=quantity,  
                    side=trade_side, 
                    type='market',
                    time_in_force='day' 
                )
                
                # 📥 NEW AUTOMATED AUDIT ENTRY
                log_trade_to_csv(ticker, trade_side, quantity, current_price, total_cost, probability, market_regime)
                print(f" ✅ Order routed and documented into local trade ledger.")
            else:
                print(f" ⚠️ Skipped: Target allocation (${allocation_per_stock:.2f}) is too low for 1 whole share of {ticker} (@ ${current_price:.2f}).")
                
        except Exception as e:
            print(f" ❌ Order Execution Failed for {ticker}: {e}")

    print("\n🎉 Order transmission loop complete. Daily audit file 'trade_execution_log.csv' successfully populated.")

# ==========================================
# MAIN EXECUTION GATEWAY
# ==========================================
if __name__ == "__main__":
    # 1. Fire off your Alpaca trading loop
    run_order_execution()
    
    # 2. Automatically push the results to your live GitHub Pages dashboard
    push_updates_to_github()