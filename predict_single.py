import os
import sys
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import yfinance as yf

# Suppress annoying dataframe warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# =========================================================================
# 🎛️ CONFIGURATION SWITCH
# Set this to True to match your current stock_features_master.csv bug.
# Set this to False when you fix your feature engine to track real macro returns.
# =========================================================================
USE_CSV_ZERO_MACRO_BUG = True 

def calculate_live_features_bulletproof(ticker):
    """Extracts features for any stock completely avoiding cross-asset date synchronization flaws."""
    try:
        print(f"📡 Fetching live market data for {ticker}...")
        stock = yf.Ticker(ticker)
        # Pull 2 years to give EMAs plenty of runway to normalize
        df_stock = stock.history(period="2y", interval="1d").ffill()
        
        if df_stock.empty or len(df_stock) < 35:
            print(f"❌ Error: Insufficient live data found for {ticker}.")
            return None
            
        print("🌍 Gathering global macro overlays...")
        spy = yf.Ticker("SPY").history(period="2y", interval="1d").ffill()
        vix = yf.Ticker("^VIX").history(period="2y", interval="1d").ffill()
        usd = yf.Ticker("UUP").history(period="2y", interval="1d").ffill()
        
        if spy.empty or vix.empty or usd.empty:
            print("❌ Error: Missing global macro structural data feeds.")
            return None

        # --- 1. Align Everything by Dropping Today's Unfinished Live Candle ---
        spy = spy.iloc[:-1]
        vix = vix.iloc[:-1]
        usd = usd.iloc[:-1]
        df_stock = df_stock.iloc[:-1]

        # --- 2. Standardized Feature Calculations ---
        if USE_CSV_ZERO_MACRO_BUG:
            # Replicate the master CSV profile exactly
            market_ret_1d = 0.0
            vix_ret_1d = 0.0
            usd_ret_1d = 0.0
            market_ret_5d_sum = 0.0
        else:
            # Real production calculations
            market_ret_1d = spy['Close'].pct_change().iloc[-1]
            vix_ret_1d = vix['Close'].pct_change().iloc[-1]
            usd_ret_1d = usd['Close'].pct_change().iloc[-1]
            market_ret_5d_sum = spy['Close'].pct_change().iloc[-5:].sum()
        
        # Stock-specific actions stay identical regardless
        stock_ret_1d = df_stock['Close'].pct_change().iloc[-1]
        stock_ret_5d = df_stock['Close'].pct_change(5).iloc[-1]
        relative_perf_5d = stock_ret_5d - market_ret_5d_sum
        
        # --- 3. Compute Inline MACD Histogram ---
        ema12 = df_stock['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df_stock['Close'].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        macd_hist = (macd - macd_signal).iloc[-1]
        
        # --- 4. Structural Verification Matrix Layout ---
        feature_columns = [
            'Market_Ret_1d', 'VIX_Ret_1d', 'USD_Ret_1d', 
            'Stock_Ret_1d', 'Stock_Ret_5d', 'Relative_Performance_5d', 'MACD_Hist'
        ]
        
        df_features = pd.DataFrame([[
            market_ret_1d,
            vix_ret_1d,
            usd_ret_1d,
            stock_ret_1d,
            stock_ret_5d,
            relative_perf_5d,
            macd_hist
        ]], columns=feature_columns)
        
        # --- 5. DIAGNOSTIC: CSV vs LIVE FEATURE MATCH ---
        try:
            matrix_path = "stock_features_master.csv"
            if os.path.exists(matrix_path):
                df_csv = pd.read_csv(matrix_path)
                
                if 'Ticker' in df_csv.columns:
                    csv_row = df_csv[df_csv['Ticker'] == ticker]
                else:
                    csv_row = df_csv[df_csv.index == ticker]
                
                if not csv_row.empty:
                    csv_row = csv_row.iloc[-1]
                    print("\n" + "!"*65)
                    print(f"🔍 MATH COLLISION DETECTOR FOR {ticker}")
                    print("!"*65)
                    print(f"{'Feature':<25} | {'CSV Master':<15} | {'Live Script':<15}")
                    print("-" * 65)
                    
                    live_vals = [
                        market_ret_1d, vix_ret_1d, usd_ret_1d, 
                        stock_ret_1d, stock_ret_5d, relative_perf_5d, macd_hist
                    ]
                    
                    for feat, l_val in zip(feature_columns, live_vals):
                        if feat in csv_row:
                            c_val = float(csv_row[feat])
                            match = "✅ MATCH" if abs(c_val - l_val) < 0.0001 else "❌ MISMATCH"
                            print(f"{feat:<25} | {c_val:<15.6f} | {l_val:<15.6f} {match}")
                        else:
                            print(f"{feat:<25} | {'Missing in CSV':<15} | {l_val:<15.6f} ❌")
                    print("!"*65 + "\n")
        except Exception:
            pass
            
        return df_features, df_stock['Close'].iloc[-1]
        
    except Exception as e:
        print(f"❌ Structural extraction failure: {e}")
        return None

def main():
    print("==============================================================")
    print("🤖 XGBOOST UNIVERSAL LIVE INFERENCE TERMINAL")
    print("==============================================================")
    
    scaler_path = "portfolio_scaler.pkl"
    model_path = "portfolio_xgb_model.json"
    
    if not os.path.exists(scaler_path) or not os.path.exists(model_path):
        print("[CRITICAL] Missing pipeline pkl/json model configurations.")
        return

    print("🧠 Loading model architectures into RAM...")
    scaler = joblib.load(scaler_path)
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    print("✅ Inference matrix ready.\n")

    while True:
        ticker_input = input("🎯 Enter ANY stock ticker (or type 'exit' to quit): ").strip().upper()
        
        if ticker_input == 'EXIT':
            print("👋 Closed live terminal.")
            break
            
        if not ticker_input:
            continue
            
        ticker = ticker_input.replace('.', '-')
        
        result = calculate_live_features_bulletproof(ticker)
        if result is None:
            print("--------------------------------------------------------------")
            continue
            
        df_features, last_price = result
        
        # Vector transformation through the pre-fitted pipeline shapes
        X_scaled = scaler.transform(df_features)
        probability = model.predict_proba(X_scaled)[0, 1]
        
        print("\n" + "-"*45)
        print(f"📊 LIVE ANALYSIS FOR: {ticker_input}")
        print(f"💵 Current Close Price: ${last_price:.2f}")
        print(f"📈 AI Breakout Confidence: {probability * 100:.2f}%")
        
        if probability >= 0.65:
            print("🟢 Status: STRONG BREAKOUT CONVICTION (Passes 65% Threshold)")
        elif probability >= 0.50:
            print("🟡 Status: MILD UPWARD BIAS (Leans Positive, Low Edge)")
        else:
            print("🔴 Status: NO BREAKOUT SIGNAL DETECTED (High Noise / Conflicted)")
        print("-"*45 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Closed live terminal.")