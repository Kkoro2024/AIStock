import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

def train_portfolio_model():
    print("==================================================")
    print("🧠 STAGE 4: TRAINING INSTITUTIONAL QUANT MODEL")
    print("==================================================")
    
    print("Loading engineered master feature matrix...")
    df = pd.read_csv("stock_features_master.csv", index_col='Date', parse_dates=True)
    df = df.sort_index()
    
    # Define our updated institutional feature columns
    feature_cols = [
        'Market_Ret_1d', 'VIX_Ret_1d', 'USD_Ret_1d', 
        'Stock_Ret_1d', 'Stock_Ret_5d', 'Relative_Performance_5d', 
        'Stock_Ret_1d_Rank', 'Stock_Ret_5d_Rank', 'MACD_Hist'
    ]
    
    X = df[feature_cols].values
    y = df['Target_Alpha_1d'].values
    
    # Chronological Train/Test Split (80% Train / 20% Test) to protect timeline purity
    split_idx = int(len(df) * 0.80)
    
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Training rows: {X_train.shape[0]} | Testing rows: {X_test.shape[0]}")
    
    # Scale features
    print("Fitting feature scaling framework...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler immediately for live use
    joblib.dump(scaler, "portfolio_scaler.pkl")
    print("✅ Saved 'portfolio_scaler.pkl'")
    
    # Initialize XGBoost with standard institutional regularization parameters
    print("Training XGBoost Classifier...")
    model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate Out-of-Sample Accuracy
    accuracy = model.score(X_test_scaled, y_test)
    print(f"\n📊 Out-of-Sample (Test Set) Accuracy: {accuracy * 100:.2f}%")
    
    # Save the updated model file
    model.save_model("portfolio_xgb_model.json")
    print("✅ Saved updated 'portfolio_xgb_model.json'")
    
    # Display Feature Importance
    importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\n💡 Feature Importance Breakdown:")
    print(importance)

if __name__ == "__main__":
    train_portfolio_model()