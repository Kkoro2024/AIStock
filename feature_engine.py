import pandas as pd
import numpy as np

def engineer_stock_features():
    print("==================================================")
    print("🛠️ STAGE 3: MULTI-STOCK FEATURE ENGINE")
    print("==================================================")
    
    print("Loading raw stock matrix...")
    df = pd.read_csv("raw_stock_matrix.csv", index_col='Date', parse_dates=True)
    
    # Sort chronologically to make sure rolling math works perfectly
    df = df.sort_index()

    # --- 1. Base Market / Macro Features (Shifted to prevent look-ahead bias) ---
    print("Calculating macro benchmark returns...")
    df['Market_Ret_1d'] = df['Market_Price'].pct_change().shift(1)
    df['VIX_Ret_1d'] = df['VIX_Price'].pct_change().shift(1)
    df['USD_Ret_1d'] = df['USD_Price'].pct_change().shift(1)

    # --- 2. Isolated Stock Features (Using groupby to isolate each ticker) ---
    print("Calculating group-isolated technical indicators...")
    
    df['Stock_Ret_1d'] = df.groupby('Ticker')['Stock_Price'].transform(lambda x: x.pct_change().shift(1))
    df['Stock_Ret_5d'] = df.groupby('Ticker')['Stock_Price'].transform(lambda x: x.pct_change(periods=5).shift(1))
    
    daily_market_returns = df.groupby(level='Date')['Market_Ret_1d'].first()
    rolling_market_5d = daily_market_returns.rolling(5).sum()
    df['Relative_Performance_5d'] = df['Stock_Ret_5d'] - df.index.map(rolling_market_5d)

    # --- NEW: CROSS-SECTIONAL RANKING FEATURES ---
    print("Calculating daily cross-sectional momentum ranks...")
    # Rank every stock against its peers on each unique calendar day (pct=True scales it from 0.0 to 1.0)
    df['Stock_Ret_1d_Rank'] = df.groupby(level='Date')['Stock_Ret_1d'].rank(pct=True)
    df['Stock_Ret_5d_Rank'] = df.groupby(level='Date')['Stock_Ret_5d'].rank(pct=True)

    # --- MACD HISTOGRAM (Calculated on lagged prices to prevent leakage) ---
    print("Calculating MACD histogram on lagged historical matrix...")
    lagged_price = df.groupby('Ticker')['Stock_Price'].shift(1)
    
    ema12 = lagged_price.groupby(df['Ticker']).transform(lambda x: x.ewm(span=12, adjust=False).mean())
    ema26 = lagged_price.groupby(df['Ticker']).transform(lambda x: x.ewm(span=26, adjust=False).mean())
    macd = ema12 - ema26
    
    temp_df = pd.DataFrame({'Ticker': df['Ticker'], 'macd': macd}, index=df.index)
    macd_signal = temp_df.groupby('Ticker')['macd'].transform(lambda x: x.ewm(span=9, adjust=False).mean())
    
    df['MACD_Hist'] = macd - macd_signal

    # --- 3. THE UNIVERSAL TARGET (Institutional Cross-Sectional Alpha) ---
    print("Generating forward-shifted target variables...")
    tomorrow_price = df.groupby('Ticker')['Stock_Price'].shift(-1)
    future_return = (tomorrow_price / df['Stock_Price']) - 1
    df['Future_Return'] = future_return
    
    df['Market_Mean_Future_Return'] = df.groupby(level='Date')['Future_Return'].transform('mean')
    df['Target_Alpha_1d'] = (df['Future_Return'] > df['Market_Mean_Future_Return']).astype(int)
    df.loc[tomorrow_price.isna(), 'Target_Alpha_1d'] = np.nan
    
    df = df.drop(columns=['Future_Return', 'Market_Mean_Future_Return'])

    # --- 4. Cleanup and Save ---
    df = df.drop(columns=['Stock_Price', 'Market_Price', 'USD_Price', 'VIX_Price'])
    df = df.dropna()

    df.to_csv("stock_features_master.csv")
    print(f"\n✅ SUCCESS: Feature matrix built! Final Shape: {df.shape}")
    
    balance = df['Target_Alpha_1d'].value_counts(normalize=True) * 100
    print(f"Target Distribution:\n{balance.round(1)}")

if __name__ == "__main__":
    engineer_stock_features()