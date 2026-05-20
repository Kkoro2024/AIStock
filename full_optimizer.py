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
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from joblib import Parallel, delayed

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.filterwarnings('ignore', message=".*A value is trying to be set on a copy.*")

# ==========================================
# ADVANCED HYPERPARAMETER CONFIGURATION
# ==========================================
BASE_CAPITAL_PER_POSITION = 2000.00  
MIN_CAPITAL = 500.00
MAX_CAPITAL = 8000.00
SLIPPAGE_RATE = 0.001           

NUMBER_OF_TEST_DAYS = 250        
OPTIMIZER_STEPS = 450            

GLOBAL_DATA_CACHE = {}
SPY_CACHE = pd.DataFrame()

CONFIG_FILE = "alpaca_config.json"
with open(CONFIG_FILE, "r") as f:
    config = json.load(f)

api = tradeapi.REST(
    config.get("ALPACA_API_KEY", "").strip(), 
    config.get("ALPACA_SECRET_KEY", "").strip(), 
    config.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").strip(), 
    api_version='v2'
)

def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        table = pd.read_html(response.read())
    return table[0]["Symbol"].str.replace('-', '.', regex=False).tolist()

# ==========================================
# UPDATED DATA DOWNLOADER (CLEAN PROGRESS LOGGING)
# ==========================================
def bulk_download_historical_data(tickers, static_days):
    global GLOBAL_DATA_CACHE, SPY_CACHE
    print(f"\n📥 [SHIELD] Loading Local Cache for {len(static_days)} evaluation days...")
    dates_sorted = sorted([datetime.strptime(d, "%Y-%m-%d") for d in static_days])
    absolute_start = (dates_sorted[0] - timedelta(days=365)).strftime("%Y-%m-%d") 
    absolute_end = (dates_sorted[-1] + timedelta(days=2)).strftime("%Y-%m-%d")
    
    try:
        SPY_CACHE = api.get_bars("SPY", timeframe="1Day", start=absolute_start, end=absolute_end).df
    except Exception as e:
        print(f"[FATAL] SPY Failure: {e}")
        return False

    ticker_chunk_size = 50  
    total_tickers = len(tickers)
    
    for t_idx in range(0, total_tickers, ticker_chunk_size):
        chunk = tickers[t_idx:t_idx + ticker_chunk_size]
        current_upper_bound = min(t_idx + ticker_chunk_size, total_tickers)
        
        # 🟢 CLEAN PROGRESS PRINT: "fetching assets X of Y"
        print(f"[PROGRESS] Fetching assets {current_upper_bound} of {total_tickers}...")
        
        try:
            chunk_bars = api.get_bars(chunk, timeframe="1Day", start=absolute_start, end=absolute_end).df
            if not chunk_bars.empty:
                chunk_bars = chunk_bars.reset_index()
                for symbol in chunk_bars['symbol'].unique():
                    GLOBAL_DATA_CACHE[symbol] = chunk_bars[chunk_bars['symbol'] == symbol].copy()
        except Exception as e: 
            print(f"[WARN] Error pulling chunk: {e}")
            
        time.sleep(2.0) 
        
    print(f"✅ [SUCCESS] Local memory cache built for {len(GLOBAL_DATA_CACHE)} assets. Network disconnected.")
    return True

# ==========================================
# 6-VARIABLE PARALLEL BACKTEST ENGINE
# ==========================================
def run_cached_backtest(target_date_str, tickers, current_z, target_vol, max_mult, stop_loss, spy_fast_sma, spy_slow_sma):
    global GLOBAL_DATA_CACHE, SPY_CACHE
    target_date_iso = target_date_str 
    if SPY_CACHE.empty: return None

    spy_df = SPY_CACHE.copy()
    spy_df.index = pd.to_datetime(spy_df.index).strftime('%Y-%m-%d')
    spy_historical = spy_df[spy_df.index < target_date_iso].copy()
    
    if len(spy_historical) < spy_slow_sma:
        shorts_enabled = True
    else:
        spy_historical.loc[:, 'SMA_Fast'] = spy_historical['close'].rolling(window=int(spy_fast_sma)).mean()
        spy_historical.loc[:, 'SMA_Slow'] = spy_historical['close'].rolling(window=int(spy_slow_sma)).mean()
        shorts_enabled = float(spy_historical['SMA_Fast'].to_numpy()[-1]) < float(spy_historical['SMA_Slow'].to_numpy()[-1])

    all_stocks_data = []
    for ticker in tickers:
        if ticker not in GLOBAL_DATA_CACHE: continue
        try:
            stock_df = GLOBAL_DATA_CACHE[ticker].copy()
            stock_df['date_str'] = pd.to_datetime(stock_df['timestamp']).dt.strftime('%Y-%m-%d')
            historical_rows = stock_df[stock_df['date_str'] < target_date_iso].copy()
            target_day_rows = stock_df[stock_df['date_str'] == target_date_iso]
            
            if historical_rows.empty or len(historical_rows) < 12 or target_day_rows.empty: continue
                
            prior_close = float(historical_rows['close'].iloc[-1])
            start_window_close = float(historical_rows['close'].iloc[-12])
            trend_score = ((prior_close - start_window_close) / start_window_close) * 100
            
            daily_returns = historical_rows['close'].tail(12).pct_change().dropna()
            volatility = float(daily_returns.std())
            if pd.isna(volatility) or volatility <= 0.001: volatility = target_vol 
            
            all_stocks_data.append({
                "Ticker": ticker, "Trend_Score": trend_score, "Volatility": volatility,
                "Open": float(target_day_rows['open'].iloc[0]), 
                "High": float(target_day_rows['high'].iloc[0]),  
                "Low": float(target_day_rows['low'].iloc[0]),    
                "Close": float(target_day_rows['close'].iloc[0])
            })
        except Exception: continue

    df_universe = pd.DataFrame(all_stocks_data)
    if df_universe.empty or len(df_universe) < 50: return None

    df_universe["Z_Score"] = (df_universe["Trend_Score"] - df_universe["Trend_Score"].mean()) / df_universe["Trend_Score"].std()
    longs_df = df_universe[df_universe["Z_Score"] >= current_z].copy()
    shorts_df = df_universe[df_universe["Z_Score"] <= -current_z].copy() if shorts_enabled else pd.DataFrame(columns=df_universe.columns)

    day_pnl = 0.0
    
    def calc_pnl(df):
        confidence_multiplier = (df["Z_Score"].abs() / current_z).clip(upper=max_mult)
        volatility_multiplier = target_vol / df["Volatility"]
        df["Capital"] = (BASE_CAPITAL_PER_POSITION * confidence_multiplier * volatility_multiplier).clip(lower=MIN_CAPITAL, upper=MAX_CAPITAL)
        df["Shares"] = df["Capital"] / df["Open"]
        
        pnl_sum = 0.0
        for _, row in df.iterrows():
            shares, op, cl, hi, lo = row["Shares"], row["Open"], row["Close"], row["High"], row["Low"]
            if df.name == 'long':
                stop_price = op * (1.0 - stop_loss)
                profit = (stop_price - op) * shares if lo <= stop_price else (cl - op) * shares
            else:
                stop_price = op * (1.0 + stop_loss)
                profit = (op - stop_price) * shares if hi >= stop_price else (op - cl) * shares
                    
            pnl_sum += (profit - ((op + cl) * shares * SLIPPAGE_RATE))
        return pnl_sum

    if not longs_df.empty: longs_df.name = 'long'; day_pnl += calc_pnl(longs_df)
    if not shorts_df.empty: shorts_df.name = 'short'; day_pnl += calc_pnl(shorts_df)
    return day_pnl

def generate_macro_testing_window():
    start_date, end_date = datetime.strptime("2022-01-01", "%Y-%m-%d"), datetime.strptime("2026-03-01", "%Y-%m-%d")
    total_days = (end_date - start_date).days
    static_days = set()
    while len(static_days) < NUMBER_OF_TEST_DAYS:
        r_date = start_date + timedelta(days=random.randint(0, total_days))
        if r_date.weekday() < 5 and not any(r_date.strftime("%m-%d") == h for h in ["-01-01", "-07-04", "-12-25"]):
            static_days.add(r_date.strftime("%Y-%m-%d"))
    return sorted(list(static_days))

# ==========================================
# EXECUTION INTERFACE
# ==========================================
if __name__ == "__main__":
    static_ticker_universe = get_sp500_tickers()
    fixed_historical_window = generate_macro_testing_window()
    
    if bulk_download_historical_data(static_ticker_universe, fixed_historical_window):
        space = [
            Real(1.2, 3.3, name='z_score'),          
            Real(0.01, 0.04, name='target_vol'),     
            Real(1.0, 4.0, name='max_mult'),
            Real(0.005, 0.04, name='stop_loss'),         
            Integer(10, 50, name='spy_fast_sma'),        
            Integer(50, 200, name='spy_slow_sma')        
        ]

        @use_named_args(space)
        def objective(z_score, target_vol, max_mult, stop_loss, spy_fast_sma, spy_slow_sma):
            step_start_time = time.time()
            session_returns = Parallel(n_jobs=-1)(
                delayed(run_cached_backtest)(
                    test_day, static_ticker_universe, z_score, target_vol, max_mult, 
                    stop_loss, spy_fast_sma, spy_slow_sma
                )
                for test_day in fixed_historical_window
            )
            session_returns = [pnl for pnl in session_returns if pnl is not None]
                    
            if len(session_returns) > 0:
                mean_pnl = np.mean(session_returns)
                stdev = np.std(session_returns)
                performance_score = mean_pnl - (0.1 * stdev)
            else:
                mean_pnl, stdev, performance_score = -5000.0, 1.0, -6000.0
                
            print(f"📊 6D Evaluated -> Z: {z_score:.2f} | Vol: {target_vol:.3f} | Mult: {max_mult:.1f}x | Stop: {stop_loss*100:.2f}% | Fast: {int(spy_fast_sma)}d | Slow: {int(spy_slow_sma)}d | Mean PnL: ${mean_pnl:+.2f} | Score: {performance_score:,.2f} | Step Time: {time.time() - step_start_time:.1f}s")
            return -performance_score

        print(f"\n🚀 Commencing Advanced 6D Multi-Threaded Structural Machine Learning Sequence...")
        result = gp_minimize(objective, space, n_calls=OPTIMIZER_STEPS, random_state=42)
        
        print("\n" + "="*60)
        print(f"🥇 MASTER 6D OPTIMIZATION COMPLETE")
        print(f"Optimal Z-Score Target: {result.x[0]:.4f}")
        print(f"Optimal Target Volatility: {result.x[1]:.4f}")
        print(f"Optimal Max Multiplier: {result.x[2]:.2f}x")
        print(f"Optimal Stop Loss: {result.x[3]*100:.3f}%")
        print(f"Optimal SPY Fast SMA: {int(result.x[4])} days")
        print(f"Optimal SPY Slow SMA: {int(result.x[5])} days")
        print("="*60)