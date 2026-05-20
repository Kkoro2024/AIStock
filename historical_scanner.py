import pandas as pd
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame
import urllib.request
import os
import random
import json
import numpy as np
import time
import warnings
from datetime import datetime, timedelta

warnings.simplefilter(action='ignore', category=FutureWarning)

# ==================================================================
# 6D ARCHITECTURE PARAMETER SETTINGS (OPTIMIZER ALIGNED)
# ==================================================================
REQUIRED_Z_SCORE         = 3.1052   # Noise threshold anchor
TARGET_DAILY_VOLATILITY  = 0.0100   # Standard deviation volatility cap (1.00%)
MAX_MULT                 = 1.00     # Uniform weight capital scaling multiplier
STOP_LOSS_PCT            = 0.0050   # 0.50% Rigid micro-shield capital preservation line
FAST_SMA_DAYS            = 50       # High-performance short-term baseline
SLOW_SMA_DAYS            = 92       # Macro structural trend ceiling

BASE_CAPITAL_PER_POSITION = 2000.00  
MAX_CAPITAL_PER_POSITION  = 6000.00
MIN_CAPITAL_PER_POSITION  = 500.00
SLIPPAGE_RATE             = 0.0002  # Realistic large-cap execution friction
NUMBER_OF_RUNS            = 100          

# ==========================================
# 1. LOAD ALPACA CONFIGURATION & INITIALIZE
# ==========================================
CONFIG_FILE = "alpaca_config.json"
with open(CONFIG_FILE, "r") as f:
    config = json.load(f)

ALPACA_API_KEY = config.get("ALPACA_API_KEY", "").strip()
ALPACA_SECRET_KEY = config.get("ALPACA_SECRET_KEY", "").strip()
BASE_URL = config.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").strip()
if BASE_URL.endswith("/v2"):
    BASE_URL = BASE_URL.replace("/v2", "")

api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, BASE_URL, api_version='v2')

# ==========================================
# 2. FETCH S&P 500 UNIVERSE LOGIC
# ==========================================
def get_sp500_tickers():
    print("[INFO] Fetching S&P 500 ticker list from Wikipedia...")
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        table = pd.read_html(response.read())
    df = table[0]
    tickers = df["Symbol"].str.replace('-', '.', regex=False).tolist()
    return tickers

# ==========================================
# 3. UNIFORM RISK PARITY SIZING FUNCTION
# ==========================================
def calculate_dynamic_capital(df):
    """Applies risk parity inverse to volatility while enforcing flat, uniform weight scaling."""
    # Conviction booster multiplier completely disabled (MAX_MULT = 1.00x)
    confidence_multiplier = 1.00
    
    # Risk Parity Allocation (Higher volatility = scaled down capital)
    volatility_multiplier = TARGET_DAILY_VOLATILITY / df["Volatility"]
    
    raw_capital = BASE_CAPITAL_PER_POSITION * confidence_multiplier * volatility_multiplier
    return raw_capital.clip(lower=MIN_CAPITAL_PER_POSITION, upper=MAX_CAPITAL_PER_POSITION)

# ==========================================
# 4. TWO-TAILED ENGINE REWIRED FOR 6D METRICS
# ==========================================
def run_two_tailed_backtest(target_date_str, tickers):
    target_date = datetime.strptime(target_date_str, "%Y-%m-%d")
    
    macro_start = (target_date - timedelta(days=300)).strftime("%Y-%m-%d")
    window_start = (target_date - timedelta(days=12)).strftime("%Y-%m-%d")
    window_end = (target_date + timedelta(days=1)).strftime("%Y-%m-%d")
    
    print(f"\n[DATA] Querying Alpaca REST API for date: {target_date_str}")
    
    try:
        spy_req = api.get_bars("SPY", TimeFrame.Day, start=macro_start, end=window_end)
        spy_bars = spy_req.df if hasattr(spy_req, 'df') else pd.DataFrame()
    except Exception as e:
        print(f"[ERROR] SPY macro data fetch failed: {e}")
        return None

    if spy_bars.empty:
        print("[ERROR] SPY returned no data. Skipping date.")
        return None

    spy_bars.index = pd.to_datetime(spy_bars.index).strftime('%Y-%m-%d')

    all_raw_bars = []
    chunk_size = 100
    
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        try:
            bars = api.get_bars(chunk, TimeFrame.Day, start=window_start, end=window_end)
            if hasattr(bars, 'df') and not bars.df.empty:
                all_raw_bars.append(bars.df.reset_index())
        except Exception as e:
            pass
        time.sleep(0.4) 
        
    if not all_raw_bars:
        print("[ERROR] No S&P 500 data returned. Skipping date.")
        return None
        
    raw_bars = pd.concat(all_raw_bars, ignore_index=True)

    # --- 50/92 SMA MOMENTUM REGIME FILTER ---
    spy_historical = spy_bars[spy_bars.index < target_date_str].copy()
    if len(spy_historical) < SLOW_SMA_DAYS:
        shorts_enabled = True
        market_regime = "UNDETERMINED (Insufficient Macro Data)"
    else:
        spy_historical['FAST_SMA'] = spy_historical['close'].rolling(window=FAST_SMA_DAYS).mean()
        spy_historical['SLOW_SMA'] = spy_historical['close'].rolling(window=SLOW_SMA_DAYS).mean()
        
        last_fast_sma = float(spy_historical['FAST_SMA'].iloc[-1])
        last_slow_sma = float(spy_historical['SLOW_SMA'].iloc[-1])
        
        if last_fast_sma >= last_slow_sma:
            shorts_enabled = False
            market_regime = f"🟢 BULL REGIME GERAING (Fast SMA {last_fast_sma:.2f} >= Slow SMA {last_slow_sma:.2f} | Shorts Disabled)"
        else:
            shorts_enabled = True
            market_regime = f"🔴 BEAR REGIME GEARING (Fast SMA {last_fast_sma:.2f} < Slow SMA {last_slow_sma:.2f} | Shorts Active)"
            
    print(f"[REGIME] {market_regime}")

    all_stocks_data = []
    
    for ticker in tickers:
        try:
            stock_df = raw_bars[raw_bars['symbol'] == ticker].copy()
            if len(stock_df) < 3: continue
                
            stock_df['timestamp'] = pd.to_datetime(stock_df['timestamp']).dt.strftime('%Y-%m-%d')
            stock_df = stock_df.set_index('timestamp')
            
            historical_rows = stock_df[stock_df.index < target_date_str]
            target_day_rows = stock_df[stock_df.index == target_date_str]
            
            if historical_rows.empty or target_day_rows.empty: continue
            
            daily_returns = historical_rows['close'].pct_change().dropna()
            volatility = float(daily_returns.std())
            
            if pd.isna(volatility) or volatility <= 0.001:
                volatility = TARGET_DAILY_VOLATILITY 
                
            prior_close = float(historical_rows['close'].iloc[-1])
            start_window_close = float(historical_rows['close'].iloc[0])
            trend_score = ((prior_close - start_window_close) / start_window_close) * 100
            
            open_price = float(target_day_rows['open'].iloc[0])
            close_price = float(target_day_rows['close'].iloc[0])
            high_price = float(target_day_rows['high'].iloc[0])
            low_price = float(target_day_rows['low'].iloc[0])
            
            all_stocks_data.append({
                "Ticker": ticker,
                "Trend_Score": trend_score,
                "Volatility": volatility,
                "Open": open_price,
                "Close": close_price,
                "High": high_price,
                "Low": low_price
            })
        except Exception:
            continue

    df_universe = pd.DataFrame(all_stocks_data)
    if df_universe.empty or len(df_universe) < 50:
        print("[WARN] Insufficient valid assets. Skipping date.")
        return None

    mean_trend = df_universe["Trend_Score"].mean()
    std_trend = df_universe["Trend_Score"].std()

    df_universe["Z_Score"] = (df_universe["Trend_Score"] - mean_trend) / std_trend
    
    longs_df = df_universe[df_universe["Z_Score"] >= REQUIRED_Z_SCORE].copy()
    shorts_df = df_universe[df_universe["Z_Score"] <= -REQUIRED_Z_SCORE].copy() if shorts_enabled else pd.DataFrame(columns=df_universe.columns)

    # --- INTRADAY RISK ENGINE MODELING (LONGS) ---
    if not longs_df.empty:
        longs_df["Capital_Allocated"] = calculate_dynamic_capital(longs_df)
        longs_df["Shares"] = longs_df["Capital_Allocated"] // longs_df["Open"]
        
        # Simulated Stop-Loss Engine
        longs_df["Exit_Price"] = np.where(
            longs_df["Low"] <= longs_df["Open"] * (1 - STOP_LOSS_PCT),
            longs_df["Open"] * (1 - STOP_LOSS_PCT),
            longs_df["Close"]
        )
        longs_df["Is_Stopped"] = longs_df["Low"] <= longs_df["Open"] * (1 - STOP_LOSS_PCT)
        
        longs_df["Gross_Profit"] = (longs_df["Exit_Price"] - longs_df["Open"]) * longs_df["Shares"]
        longs_df["Slippage_Cost"] = (longs_df["Open"] + longs_df["Exit_Price"]) * longs_df["Shares"] * SLIPPAGE_RATE
        longs_df["Dollar_Profit"] = longs_df["Gross_Profit"] - longs_df["Slippage_Cost"]
        longs_df["Actual_Return %"] = (longs_df["Dollar_Profit"] / longs_df["Capital_Allocated"]) * 100
    else:
        longs_df["Dollar_Profit"] = pd.Series(dtype='float64')

    # --- INTRADAY RISK ENGINE MODELING (SHORTS) ---
    if not shorts_df.empty:
        shorts_df["Capital_Allocated"] = calculate_dynamic_capital(shorts_df)
        shorts_df["Shares"] = shorts_df["Capital_Allocated"] // shorts_df["Open"]
        
        # Simulated Stop-Loss Engine
        shorts_df["Exit_Price"] = np.where(
            shorts_df["High"] >= shorts_df["Open"] * (1 + STOP_LOSS_PCT),
            shorts_df["Open"] * (1 + STOP_LOSS_PCT),
            shorts_df["Close"]
        )
        shorts_df["Is_Stopped"] = shorts_df["High"] >= shorts_df["Open"] * (1 + STOP_LOSS_PCT)
        
        shorts_df["Gross_Profit"] = (shorts_df["Open"] - shorts_df["Exit_Price"]) * shorts_df["Shares"]
        shorts_df["Slippage_Cost"] = (shorts_df["Open"] + shorts_df["Exit_Price"]) * shorts_df["Shares"] * SLIPPAGE_RATE
        shorts_df["Dollar_Profit"] = shorts_df["Gross_Profit"] - shorts_df["Slippage_Cost"]
        shorts_df["Actual_Return %"] = (shorts_df["Dollar_Profit"] / shorts_df["Capital_Allocated"]) * 100
    else:
        shorts_df["Dollar_Profit"] = pd.Series(dtype='float64')

    longs_df["Direction"] = "LONG"
    shorts_df["Direction"] = "SHORT"
    final_export = pd.concat([longs_df, shorts_df])
    
    print("=========================================================================================")
    print(f"               🟢 STATISTICAL LONG SELECTIONS                      ")
    print("=========================================================================================")
    if longs_df.empty: print("No assets crossed the positive Z-threshold boundary.")
    for _, row in longs_df.sort_values(by="Z_Score", ascending=False).iterrows():
        stop_msg = "[STOP TRIGGERED]" if row['Is_Stopped'] else "[CLEAN CLOSE]"
        print(f"{row['Ticker']:<5} | Z: {row['Z_Score']:>4.2f} | Status: {stop_msg:<16} | Alloc: ${row['Capital_Allocated']:>6.0f} | Return: {row['Actual_Return %']:+6.2f}%")

    print("\n=========================================================================================")
    print(f"               🔴 STATISTICAL SHORT SELECTIONS                     ")
    print("=========================================================================================")
    if not shorts_enabled: print("Short execution BLOCKED by 50/92 SMA Trend Gearing Filter Regime.")
    elif shorts_df.empty: print("No assets crossed the negative Z-threshold boundary.")
    for _, row in shorts_df.sort_values(by="Z_Score", ascending=True).iterrows():
        stop_msg = "[STOP TRIGGERED]" if row['Is_Stopped'] else "[CLEAN CLOSE]"
        print(f"{row['Ticker']:<5} | Z: {row['Z_Score']:>5.2f} | Status: {stop_msg:<16} | Alloc: ${row['Capital_Allocated']:>6.0f} | Return: {row['Actual_Return %']:+6.2f}%")
    
    total_longs_count = len(longs_df)
    total_shorts_count = len(shorts_df)
    day_longs_pnl = longs_df["Dollar_Profit"].sum() if total_longs_count > 0 else 0.0
    day_shorts_pnl = shorts_df["Dollar_Profit"].sum() if total_shorts_count > 0 else 0.0
    
    total_capital_deployed = (longs_df["Capital_Allocated"].sum() if not longs_df.empty else 0) + (shorts_df["Capital_Allocated"].sum() if not shorts_df.empty else 0)
    net_session_pnl = day_longs_pnl + day_shorts_pnl

    print("\n==================================================================")
    print(f"                 💰 SESSION FINANCIAL ACCOUNTING                  ")
    print("==================================================================")
    print(f"  Total Capital Deployed: ${total_capital_deployed:,.2f}")
    print(f"  Long Portfolio Profit:  ${day_longs_pnl:+,.2f} ({total_longs_count} positions)")
    print(f"  Short Portfolio Profit: ${day_shorts_pnl:+,.2f} ({total_shorts_count} positions)")
    print("-" * 66)
    print(f"  NET TOTAL SESSION CASH RETURN: ${net_session_pnl:+,.2f}")
    print("==================================================================\n")

    output_folder = "Screener_Reports"
    if not os.path.exists(output_folder): os.makedirs(output_folder)

    if not final_export.empty:
        ledger_file = os.path.join(output_folder, "Master_Performance_Ledger.csv")
        ledger_data = pd.DataFrame([{
            "Scan_Date": target_date_str,
            "Execution_Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Long_Count": total_longs_count,
            "Short_Count": total_shorts_count,
            "Capital_Deployed": round(total_capital_deployed, 2),
            "Long_PnL": round(day_longs_pnl, 2),
            "Short_PnL": round(day_shorts_pnl, 2),
            "Net_Total_PnL": round(net_session_pnl, 2)
        }])
        ledger_exists = os.path.exists(ledger_file)
        ledger_data.to_csv(ledger_file, mode='a', index=False, header=not ledger_exists)

        return net_session_pnl
    return None

# ==========================================
# 5. RUNTIME AUTOMATION EXECUTIVE
# ==========================================
if __name__ == "__main__":
    static_ticker_universe = get_sp500_tickers()
    start_date = datetime.strptime("2022-01-01", "%Y-%m-%d")
    end_date = datetime.strptime("2026-03-01", "%Y-%m-%d")
    total_days = (end_date - start_date).days
    
    unique_simulation_days = set()
    session_returns = []
    
    print(f"\n[START] Commencing full simulation filter loop...")
    
    while len(session_returns) < NUMBER_OF_RUNS:
        random_datetime = start_date + timedelta(days=random.randint(0, total_days))
        date_str = random_datetime.strftime("%Y-%m-%d")
        
        if date_str in unique_simulation_days or random_datetime.weekday() >= 5:
            continue
            
        time.sleep(0.5)
        pnl = run_two_tailed_backtest(date_str, static_ticker_universe)
        
        if pnl is not None:
            unique_simulation_days.add(date_str)
            session_returns.append(pnl)
            print(f"📈 [LOGGED] {len(session_returns)} / {NUMBER_OF_RUNS} complete.")

    print("\n" + "="*50)
    print(f"🏁 [BACKTEST COMPLETE]")
    print(f"Composite Mean Return: ${np.mean(session_returns):,.2f}")
    print("="*50)