import pandas as pd
import alpaca_trade_api as tradeapi
import os
import json
import datetime
import requests

CONFIG_FILE = "alpaca_config.json"
with open(CONFIG_FILE, "r") as f:
    config = json.load(f)

# ALIGNED INTERFACE: Fixed the v2 duplication issue cleanly via strip and replace
api = tradeapi.REST(
    config.get("ALPACA_API_KEY", "").strip(),
    config.get("ALPACA_SECRET_KEY", "").strip(),
    config.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").replace("/v2", "").strip(),
    api_version='v2'
)

def sync_to_google_sheets(log_data):
    """Bypasses Google Cloud API blocks by routing metrics through a standard Google Form."""
    form_url = "https://docs.google.com/forms/d/e/1FAIpQLScUaGijHoOsB-Ve8nD6bQXfwrdI6baGcNI3_O3lu7yJh_ZgQA/formResponse"
    
    # Check if entry placeholders are mapped before attempting webhook post
    form_data = {
        "entry.672795586": log_data["Timestamp"],
        "entry.809172251": log_data["Ticker"],
        "entry.426938760": log_data["Side"],
        "entry.1089217121": log_data["Quantity"],
        "entry.314136737": log_data["Est_Entry_Price"],
        "entry.346639149": log_data["Total_Capital_Allocated"],
        "entry.1171571888": log_data["AI_Confidence"],
        "entry.714202078": log_data["Market_Regime"]
    }
    
    try:
        response = requests.post(form_url, data=form_data, timeout=5)
        if response.status_code == 200:
            print(f"📊 Cloud Update Complete: Row streamed for {log_data['Ticker']}.")
        else:
            print(f"⚠️ Form transmission returned status code: {response.status_code}")
    except Exception as e:
        print(f"❌ Webhook Sync Failure: {e}")

def log_trade_to_local_csv(ticker, side, qty, est_price, total_cost, prob, regime):
    log_file = "trade_execution_log.csv"
    log_entry = {
        "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Ticker": ticker, "Side": side.upper(), "Quantity": qty,
        "Est_Entry_Price": round(est_price, 2), "Total_Capital_Allocated": round(total_cost, 2),
        "AI_Confidence": round(prob, 2), "Market_Regime": regime
    }
    df_new = pd.DataFrame([log_entry])
    
    if os.path.exists(log_file):
        df_new.to_csv(log_file, mode='a', header=False, index=False, lineterminator='\n')
    else:
        df_new.to_csv(log_file, mode='w', header=True, index=False, lineterminator='\n')
        
    sync_to_google_sheets(log_entry)

def run_order_execution():
    print("==================================================")
    print("💼 BI-DIRECTIONAL BROKER ROUTING ENGINE")
    print("==================================================")
    
    try:
        account = api.get_account()
    except Exception as e:
        print(f"❌ Connection Failed: {e}")
        return

    available_capital = float(account.cash)
    sheet_file = 'live_portfolio_selection.csv'
    if not os.path.exists(sheet_file): 
        print("⚠️ No live selection file found.")
        return
        
    df_picks = pd.read_csv(sheet_file)
    if df_picks.empty: 
        print("⚠️ Selection file is empty.")
        return

    market_regime = df_picks['Market_Regime'].iloc[0] if 'Market_Regime' in df_picks.columns else "BULL"
    risk_scalar = 0.25 if market_regime == "BEAR" else 0.95

    # --- ALIGNED RISK CONTROL: Independent Long vs Short Capital Buckets ---
    total_cash_pool = available_capital * risk_scalar
    side_pool_max = total_cash_pool / 2  # Max 50% allocation to Buys, 50% to Shorts
    
    long_picks = df_picks[df_picks['Side'].str.upper() == 'BUY']
    short_picks = df_picks[df_picks['Side'].str.upper() == 'SELL']
    
    print(f"Total Capital Pool: ${total_cash_pool:.2f} | Side Allocation Cap: ${side_pool_max:.2f}")
    print(f"Identified Components: {len(long_picks)} Longs | {len(short_picks)} Shorts")
    
    # Calculate uniform dollar weights per active side component
    long_allocation = side_pool_max / len(long_picks) if not long_picks.empty else 0.0
    short_allocation = side_pool_max / len(short_picks) if not short_picks.empty else 0.0

    for index, row in df_picks.iterrows():
        ticker = row['Ticker']
        trade_side = row['Side'].lower()
        probability = row['AI_Probability'] * 100 if 'AI_Probability' in row else 0.0
        
        # Select target cash amount based on current trade side
        target_allocation = long_allocation if trade_side == 'buy' else short_allocation
        if target_allocation == 0.0:
            continue
            
        try:
            latest_trade = api.get_latest_trade(ticker)
            current_price = float(latest_trade.price)
            quantity = int(target_allocation // current_price)
            
            if quantity >= 1:
                total_cost = quantity * current_price
                print(f"Routing {trade_side.upper()} order for {quantity} shares of {ticker}...")
                
                try:
                    api.submit_order(
                        symbol=ticker, qty=quantity, side=trade_side,
                        type='market', time_in_force='day'
                    )
                    log_trade_to_local_csv(ticker, trade_side, quantity, current_price, total_cost, probability, market_regime)
                except Exception as order_error:
                    print(f"❌ Broker Rejected Order for {ticker} ({trade_side.upper()}): {order_error}")
            else:
                print(f"⚠️ Unit pricing exceeds individual allocation size for {ticker}. Skipped.")
        except Exception as data_error:
            print(f"❌ Market Data Parsing Failure for {ticker}: {data_error}")

if __name__ == "__main__":
    run_order_execution()