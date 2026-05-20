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
# 🥇 LOCKED: The optimal Z-Score we found in the previous run
REQUIRED_Z_SCORE = 2.9222  

BASE_CAPITAL_PER_POSITION = 2000.00  
MIN_CAPITAL = 500.00
MAX_CAPITAL = 8000.00
SLIPPAGE_RATE = 0.001           
NUMBER_OF_TEST_DAYS = 50        
OPTIMIZER_STEPS = 35            # Increased to handle 2 variables

# ==========================================
# 1. LOAD ALPACA CONFIGURATION
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
# 2. FETCH S&P 500 UNIVERSE
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
# 3. DYNAMIC BACKTEST ENGINE
# ==========================================
def run_dynamic_backtest(target_date_str, tickers, target_vol, max_mult):
    target_date = datetime.strptime(target_date_str, "%Y-%m-%d")
    macro_start = (target_date - timedelta(days=300)).strftime("%Y-%m-%d")
    window_start = (target_date - timedelta(days=12)).strftime("%Y-%m-%d")
    window_end = (target_date + timedelta(days=1)).strftime("%Y-%m-%d")
    
    try:
        spy_bars = api.get_bars("SPY", timeframe="1Day", start=macro_start, end=window_end).df
        
        # Chunk requests to prevent Alpaca rate limits
        chunk_size = 100
        bars_list = []
        for i in range(0, len(tickers), chunk_size):
            chunk = tickers[i:i + chunk_size]
            chunk_bars = api.get_bars(chunk, timeframe="1Day", start=window_start, end=window_end).df
            if not chunk_bars.empty: bars_list.append(chunk_bars)
            time.sleep(0.1) 
            
        if not bars_list or spy_bars.empty: return None
        raw_bars = pd.concat(bars_list).reset_index()
    except Exception as e:
        print(f"[ERROR] API Connection Interrupted: {e}")
        return None

    # Macro Regime Switch
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
                
            # Trend Math
            prior_close = float(historical_rows['close'].iloc[-1])
            start_window_close = float(historical_rows['close'].iloc[0])
            trend_score = ((prior_close - start_window_close) / start_window_close) * 100
            
            # Volatility Math (Standard Dev of historical daily % returns)
            daily_returns = historical_rows['close'].pct_change().dropna()
            volatility = float(daily_returns.std())
            if pd.isna(volatility) or volatility <= 0.001: volatility = target_vol 
            
            all_stocks_data.append({
                "Ticker": ticker, "Trend_Score": trend_score, "Volatility": volatility,
                "Open": float(target_day_rows['open'].iloc[0]), "Close": float(target_day_rows['close'].iloc[0])
            })
        except Exception:
            continue

    df_universe = pd.DataFrame(all_stocks_data)
    if df_universe.empty or len(df_universe) < 50: return None

    df_universe["Z_Score"] = (df_universe["Trend_Score"] - df_universe["Trend_Score"].mean()) / df_universe["Trend_Score"].std()
    
    longs_df = df_universe[df_universe["Z_Score"] >= REQUIRED_Z_SCORE].copy()
    shorts_df = df_universe[df_universe["Z_Score"] <= -REQUIRED_Z_SCORE].copy() if shorts_enabled else pd.DataFrame(columns=df_universe.columns)

    day_pnl = 0.0
    
    # 🥇 The Dynamic Capital Allocation Formula
    def calc_pnl(df):
        confidence_multiplier = (df["Z_Score"].abs() / REQUIRED_Z_SCORE).clip(upper=max_mult)
        volatility_multiplier = target_vol / df["Volatility"]
        
        df["Capital"] = (BASE_CAPITAL_PER_POSITION * confidence_multiplier * volatility_multiplier).clip(lower=MIN_CAPITAL, upper=MAX_CAPITAL)
        df["Shares"] = df["Capital"] / df["Open"]
        df["Profit"] = (df["Close"] - df["Open"]) * df["Shares"] if df.name == 'long' else (df["Open"] - df["Close"]) * df["Shares"]
        df["Slippage"] = (df["Open"] + df["Close"]) * df["Shares"] * SLIPPAGE_RATE
        return (df["Profit"] - df["Slippage"]).sum()

    if not longs_df.empty:
        longs_df.name = 'long'
        day_pnl += calc_pnl(longs_df)

    if not shorts_df.empty:
        shorts_df.name = 'short'
        day_pnl += calc_pnl(shorts_df)

    return day_pnl

# ==========================================
# 4. RUNTIME AUTOMATION EXECUTIVE
# ==========================================
def generate_static_testing_window():
    print(f"[INIT] Creating a locked evaluation matrix of {NUMBER_OF_TEST_DAYS} days...")
    start_date = datetime.strptime("2022-01-01", "%Y-%m-%d")
    end_date = datetime.strptime("2026-03-01", "%Y-%m-%d")
    total_days = (end_date - start_date).days
    
    static_days = set()
    while len(static_days) < NUMBER_OF_TEST_DAYS:
        random_datetime = start_date + timedelta(days=random.randint(0, total_days))
        if random_datetime.weekday() < 5 and not any(random_datetime.strftime("%m-%d") == h for h in ["-01-01", "-07-04", "-12-25"]):
            static_days.add(random_datetime.strftime("%Y-%m-%d"))
            
    return sorted(list(static_days))

if __name__ == "__main__":
    static_ticker_universe = get_sp500_tickers()
    fixed_historical_window = generate_static_testing_window()
    
    # 🥇 TWO-DIMENSIONAL SEARCH SPACE
    space = [
        Real(0.01, 0.05, name='target_vol'),   # Tests aiming for 1% to 5% baseline volatility
        Real(1.0, 4.0, name='max_mult')        # Tests capping multipliers between 1.0x and 4.0x
    ]

    @use_named_args(space)
    def objective(target_vol, max_mult):
        print(f"\n🌀 Engine Testing: Target Vol = {target_vol:.3f} | Max Multiplier = {max_mult:.2f}x")
        
        session_returns = []
        for test_day in fixed_historical_window:
            pnl = run_dynamic_backtest(test_day, static_ticker_universe, target_vol, max_mult)
            if pnl is not None:
                session_returns.append(pnl)
                
        if len(session_returns) > 0:
            mean_pnl = np.mean(session_returns)
            stdev = np.std(session_returns)
            # Sharpe-like objective calculation
            performance_score = mean_pnl - (0.1 * stdev)
        else:
            mean_pnl, stdev, performance_score = -5000.0, 1.0, -6000.0
            
        print(f"📊 Results -> Mean PnL: ${mean_pnl:,.2f} | StDev: ${stdev:,.2f} | Score: {performance_score:,.2f}")
        return -performance_score

    print(f"\n🚀 Commencing 2D Bayesian Search ({OPTIMIZER_STEPS} Iterations)...")
    result = gp_minimize(objective, space, n_calls=OPTIMIZER_STEPS, random_state=42)
    
    print("\n" + "="*60)
    print(f"🥇 SEARCH COMPLETE")
    print(f"Optimal Target Volatility: {result.x[0]:.4f}")
    print(f"Optimal Max Conviction Multiplier: {result.x[1]:.2f}x")
    print("="*60)