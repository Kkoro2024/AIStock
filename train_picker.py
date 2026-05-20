import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb

def train_universal_picker():
    print("==================================================")
    print("🧠 STAGE 4: TRAINING THE UNIVERSAL STOCK PICKER")
    print("==================================================")
    
    print("Loading master feature matrix...")
    df = pd.read_csv("stock_features_master.csv", index_col='Date', parse_dates=True)
    
    # Clean up the target row count and drop the non-numeric Ticker column for training
    df = df.dropna(subset=['Target_Alpha_1d'])
    tickers = df['Ticker'].values # Save this to analyze picks later
    
    feature_cols = [c for c in df.columns if c not in ['Target_Alpha_1d', 'Ticker']]
    X = df[feature_cols].values
    y = df['Target_Alpha_1d'].values
    
    # --- CHRONOLOGICAL DATA SPLIT ---
    # We find the date that marks the 80% boundary of our 5-year timeline
    unique_dates = df.index.unique().sort_values()
    split_date = unique_dates[int(len(unique_dates) * 0.8)]
    
    # Split training (past) and testing (future) based on that date
    train_mask = df.index < split_date
    test_mask = df.index >= split_date
    
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    test_tickers = tickers[test_mask]
    test_dates = df.index[test_mask]
    
    print(f"Training Matrix Size: {X_train.shape} (All stocks, first 4 years)")
    print(f"Testing Matrix Size:  {X_test.shape} (All stocks, final year)")
    
    # --- DATA SCALING ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, "portfolio_scaler.pkl")
    
    # Balance our class weights dynamically based on our 62/38 split
    neg_count = sum(y_train == 0)
    pos_count = sum(y_train == 1)
    scale_weight = neg_count / pos_count
    
    # --- MODEL TRAINING ---
    print("\nTraining Universal XGBoost Core...")
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.03,
        scale_pos_weight=scale_weight,
        random_state=42,
        eval_metric='logloss'
    )
    
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_train_scaled, y_train), (X_test_scaled, y_test)],
        verbose=False
    )
    
    # --- EVALUATING AS A PORTFOLIO SCREENER ---
    print("\n--- PERFORMANCE EVALUATION ---")
    probabilities = model.predict_proba(X_test_scaled)[:, 1]
    
    # Let's map our results back to a structured evaluation dataframe
    eval_df = pd.DataFrame({
        'Date': test_dates,
        'Ticker': test_tickers,
        'Probability': probabilities,
        'Actual_Breakout': y_test
    })
    
    # Simulate a daily portfolio picker:
    # Every day, we sort all stocks and select the TOP 5 stocks with the highest AI conviction
    print("Simulating a Daily Top-5 Stock Picker Portfolio...")
    
    total_days_tested = 0
    total_correct_picks = 0
    total_picks_made = 0
    
    for date, group in eval_df.groupby('Date'):
        if len(group) >= 5: # Make sure we have a solid basket of stocks to look at
            # Sort this specific day's stocks by AI probability
            top_5_picks = group.sort_values(by='Probability', ascending=False).head(5)
            
            total_days_tested += 1
            total_picks_made += len(top_5_picks)
            total_correct_picks += top_5_picks['Actual_Breakout'].sum()
            
            # Print a quick snapshot of a recent picking day for visual proof
            if total_days_tested == 1:
                print(f"\nExample Selection Sheet for Trading Day: {date.strftime('%Y-%m-%d')}")
                for _, row in top_5_picks.iterrows():
                    status = "✅ WIN" if row['Actual_Breakout'] == 1 else "❌ LOSS/CHOP"
                    print(f"  -> Picked: {row['Ticker']:<5} | AI Confidence: {row['Probability']*100:.1f}% | Outcome: {status}")
    
    portfolio_win_rate = (total_correct_picks / total_picks_made) * 100
    print("\n" + "="*50)
    print(f"🏆 FINAL PORTFOLIO RESULTS OVER TEST YEAR")
    print("="*50)
    print(f"Total Daily Picks Evaluated: {total_picks_made}")
    print(f"AI Selected Breakout Wins:  {int(total_correct_picks)}")
    print(f"Screener Top-5 Accuracy:     {portfolio_win_rate:.2f}%")
    print(f"Baseline Market Avg:         {pos_count / len(y_train) * 100:.2f}%")
    print("="*50)
    
    # Save model
    model.save_model("portfolio_xgb_model.json")
    print("\nModel saved successfully as 'portfolio_xgb_model.json'!")

if __name__ == "__main__":
    train_universal_picker()