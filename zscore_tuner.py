import pandas as pd
import alpaca_trade_api as tradeapi
import urllib.request
import os
import random
import json
import numpy as np
import time
import warnings
from datetime import datetime, timedelta
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

warnings.simplefilter(action='ignore', category=FutureWarning)

# ==========================================
# AUTOMATED OPTIMIZER CONFIGURATION
# ==========================================
CAPITAL_PER_POSITION = 2000.00  # Standardized capital allocation
SLIPPAGE_RATE = 0.001           # 10 bps friction penalty per trade
NUMBER_OF_TEST_DAYS = 50        # Number of static days used to evaluate each parameter
OPTIMIZER_STEPS = 25            # Total Bayesian search iterations

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
    print("[INFO] Fetching S&P 500 universe from Wikipedia...")
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        table = pd.read_html(response.read())
    df = table[0]
    tickers = df["Symbol"].str.replace('-', '.', regex=False).tolist()
    return tickers

# ==========================================
# 3. FIXED BACKTEST ENGINE ( friction & sizing accounted for)
# ==========================================
def run_fixed_backtest(target_date_str, tickers, current_z_threshold):
    target_date = datetime.strptime(target_date_str, "%Y-%m-%d")
    macro_start = (target_date - timedelta(days=300)).strftime("%Y-%m-%d")
    window_start = (target_date - timedelta(days=12)).strftime("%Y-%m-%d")
    window_end = (target_date + timedelta(days=1)).strftime("%Y-%m-%d")
    
    try:
        spy_bars = api.get_bars("SPY", timeframe="1Day", start=macro_start, end=window_end).df
        
        # 🟢 RATE LIMIT FIX: Chunk ticker requests to ensure Alpaca API stability
        chunk_size = 100
        bars_list = []
        for i in range(0, len(tickers), chunk_size):
            chunk = tickers[i:i + chunk_size]
            chunk_bars = api.get_bars(chunk, timeframe="1Day", start=window_start, end=window_end).df
            if not chunk_bars.empty:
                bars_list.append(chunk_bars)
            time.sleep(0.1) # Small pacing pause
            
        if not bars_list or spy_bars.empty: return None
        raw_bars = pd.concat(bars_list).reset_index()
    except Exception as e:
        print(f"[ERROR] API Connection Interrupted: {e}")
        return None

    # --- MACRO REGIME SWITCH ---
    spy_historical = spy_bars[spy_bars.index < target_date.strftime("%Y-%m-%d")].copy()
    if len(spy_historical) < 200:
        shorts_enabled = True
    else:
        spy_historical['SMA_200'] = spy_historical['close'].rolling(window=200).mean()
        shorts_enabled = float(spy_historical['close'].to_numpy()[-1]) < float(spy_historical['SMA_200'].to_numpy()[-1])

    all_stocks_data = []
    for ticker in tickers:
        try:
            stock_df = raw_bars[raw_bars['symbol'] == ticker].copy()
            if len(stock_df) < 3: continue
                
            stock_df['timestamp'] = pd.to_datetime(stock_df['timestamp']).dt.strftime('%Y-%m-%d')
            stock_df = stock_df.set_index('timestamp')
            
            historical_rows = stock_df[stock_df.index < target_date.strftime("%Y-%m-%d")]
            target_day_rows = stock_df[stock_df.index == target_date.strftime("%Y-%m-%d")]
            
            if historical_rows.empty or target_day_rows.empty: continue
                
            prior_close = float(historical_rows['close'].iloc[-1])
            start_window_close = float(historical_rows['close'].iloc[0])
            trend_score = ((prior_close - start_window_close) / start_window_close) * 100
            
            all_stocks_data.append({
                "Ticker": ticker, "Trend_Score": trend_score, 
                "Open": float(target_day_rows['open'].iloc[0]), "Close": float(target_day_rows['close'].iloc[0])
            })
        except Exception:
            continue

    df_universe = pd.DataFrame(all_stocks_data)
    if df_universe.empty or len(df_universe) < 50: return None

    df_universe["Z_Score"] = (df_universe["Trend_Score"] - df_universe["Trend_Score"].mean()) / df_universe["Trend_Score"].std()
    
    longs_df = df_universe[df_universe["Z_Score"] >= current_z_threshold].copy()
    shorts_df = df_universe[df_universe["Z_Score"] <= -current_z_threshold].copy() if shorts_enabled else pd.DataFrame(columns=df_universe.columns)

    day_pnl = 0.0
    if not longs_df.empty:
        longs_df["Shares"] = CAPITAL_PER_POSITION / longs_df["Open"]
        longs_df["Profit"] = (longs_df["Close"] - longs_df["Open"]) * longs_df["Shares"]
        longs_df["Slippage"] = (longs_df["Open"] + longs_df["Close"]) * longs_df["Shares"] * SLIPPAGE_RATE
        day_pnl += (longs_df["Profit"] - longs_df["Slippage"]).sum()

    if not shorts_df.empty:
        shorts_df["Shares"] = CAPITAL_PER_POSITION / shorts_df["Open"]
        shorts_df["Profit"] = (shorts_df["Open"] - shorts_df["Close"]) * shorts_df["Shares"]
        shorts_df["Slippage"] = (shorts_df["Open"] + shorts_df["Close"]) * shorts_df["Shares"] * SLIPPAGE_RATE
        day_pnl += (shorts_df["Profit"] - shorts_df["Slippage"]).sum()

    return day_pnl

# ==========================================
# 4. RUNTIME AUTOMATION EXECUTIVE
# ==========================================
def generate_static_testing_window():
    """Builds a locked, unchanging list of days to evaluate all steps fairly."""
    print(f"[INIT] Creating a locked evaluation matrix of {NUMBER_OF_TEST_DAYS} days...")
    start_date = datetime.strptime("2022-01-01", "%Y-%m-%d")
    end_date = datetime.strptime("2026-03-01", "%Y-%m-%d")
    total_days = (end_date - start_date).days
    
    static_days = set()
    while len(static_days) < NUMBER_OF_TEST_DAYS:
        random_datetime = start_date + timedelta(days=random.randint(0, total_days))
        # Skip weekends and clear holidays
        if random_datetime.weekday() < 5 and not any(random_datetime.strftime("%m-%d") == h for h in ["-01-01", "-07-04", "-12-25"]):
            static_days.add(random_datetime.strftime("%Y-%m-%d"))
            
    return sorted(list(static_days))

if __name__ == "__main__":
    static_ticker_universe = get_sp500_tickers()
    fixed_historical_window = generate_static_testing_window()
    
    # Define search boundaries for the Z-score search space
    space = [Real(1.0, 3.5, name='z_score')]

    @use_named_args(space)
    def objective(z_score):
        # Unpack from list structure
        current_z = float(z_score[0]) if isinstance(z_score, list) else float(z_score)
        print(f"\n🌀 Engine Testing Parameter Set: Z-Score = {current_z:.4f}")
        
        session_returns = []
        for test_day in fixed_historical_window:
            pnl = run_fixed_backtest(test_day, static_ticker_universe, current_z)
            if pnl is not None:
                session_returns.append(pnl)
                
        if len(session_returns) > 0:
            mean_pnl = np.mean(session_returns)
            stdev = np.std(session_returns)
            # Sharpe-like objective calculation: Reward returns, penalize high volatility
            performance_score = mean_pnl - (0.1 * stdev)
        else:
            mean_pnl, stdev, performance_score = -5000.0, 1.0, -6000.0
            
        print(f"📊 Results -> Mean PnL: ${mean_pnl:,.2f} | StDev: ${stdev:,.2f} | Objective Score: {performance_score:,.2f}")
        
        # gp_minimize always minimizes; negating the score maximizes it
        return -performance_score

    print(f"\n🚀 Commencing Scikit Bayesian Optimization Exploration ({OPTIMIZER_STEPS} Iterations)...")
    result = gp_minimize(objective, space, n_calls=OPTIMIZER_STEPS, random_state=42)
    
    print("\n" + "="*60)
    print(f"🥇 SEARCH COMPLETE")
    print(f"Optimal Mathematical Z-Score Target: {result.x[0]:.4f}")
    print("="*60)