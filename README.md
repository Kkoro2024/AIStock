# 🤖 Institutional Cross-Sectional Alpha Predictor & Balanced Trading Engine

A quantitative, bi-directional algorithmic trading system built to isolate cross-sectional market alpha, eliminate historical target leakage (look-ahead bias), and execute risk-balanced long/short portfolio rotations via the Alpaca API.

---

# 📊 Core System Architecture

The pipeline is split into isolated, sequential modules designed to mimic institutional quantitative workflows:

## 1. Universe Filtering (`get_universe.py`)
- Gathers the liquid asset universe.

## 2. Feature Engineering (`feature_engine.py`)
- Computes macro indicators, technical features, and cross-sectional ranks with a strict 1-day lag to preserve timeline purity.

## 3. Model Training (`train_model.py`)
- Trains an XGBoost Classifier on historical data using a chronological (non-random) split to prevent backtest leakage.

## 4. Live Evaluation (`live_portfolio.py`)
- Extracts the current market state, scales features, and filters out low-confidence assets using a strict 54% hybrid probability threshold.

## 5. Execution Router (`execute_trades.py`)
- Splits risk capital into independent 50% Long and 50% Short allocation buckets to maintain a balanced, directional-neutral profile.

## 6. Automation Layer (`orchestrate_cron.py` & `live_desk.py`)
- Background daemons tracking high-precision clock intervals to handle hands-free live execution.

---

# 🛠️ Installation & Setup

## 1. Prerequisites

Ensure your machine has Python 3.13+ installed.

Verify your local installation:

```powershell
python --version
```

---

## 2. Navigate to the Project Directory

```powershell
cd C:\Users\theko\Stock_AI_Project
```

---

## 3. Install Required Dependencies

```powershell
pip install pandas numpy xgboost scikit-learn alpaca-trade-api yfinance joblib requests
```

---

## 4. Configure API Credentials

Create a file named `alpaca_config.json` in the root project directory.

```json
{
  "ALPACA_API_KEY": "YOUR_LIVE_OR_PAPER_API_KEY",
  "ALPACA_SECRET_KEY": "YOUR_LIVE_OR_PAPER_SECRET_KEY",
  "ALPACA_BASE_URL": "https://paper-api.alpaca.markets",
  "NEWS_API_KEY": "YOUR_OPTIONAL_NEWS_API_KEY"
}
```

---

# 📈 Initial Model Compilation Workflow

## Step 1 — Generate Master Features

```powershell
python feature_engine.py
```

### Expected Output
- Balanced target distribution (~50/50)
- `stock_features_master.csv`

---

## Step 2 — Train the Quantitative Model

```powershell
python train_model.py
```

### Expected Output
- Test accuracy typically between `50.5%` and `54.5%`
- `portfolio_xgb_model.json`
- `portfolio_scaler.pkl`

---

# 🚀 Live Production Execution

## Terminal 1 — Launch the Clock Monitor Orchestrator

```powershell
python orchestrate_cron.py
```

Runs:
- 6:50 AM → Selection generation
- 7:15 AM → Broker order routing

---

## Terminal 2 — Launch the Risk Monitor Desk

```powershell
python live_desk.py
```

---

# 🛡️ Risk Management & Failsafes

## Dynamic Capital Sizing

Capital allocation is split symmetrically between long and short exposure.

Example:
- 18 Shorts
- 0 Longs

The router:
- Caps exposure at 50%
- Leaves remaining capital in cash
- Prevents overexposure during broad market rallies

---

## Confidence Thresholding

Assets between:
- `46.0%`
- `54.0%`

are treated as statistical noise and removed from execution.

---

## Timeline Insulation

All indicators are calculated using finalized historical closing bars only.

The system never assumes knowledge of:
- current intraday closes
- future candles
- unreleased market information

This preserves timeline integrity and prevents look-ahead bias.

---

# 📂 Example Project Structure

```text
Stock_AI_Project/
│
├── feature_engine.py
├── train_model.py
├── live_portfolio.py
├── execute_trades.py
├── orchestrate_cron.py
├── live_desk.py
├── get_universe.py
│
├── stock_features_master.csv
├── portfolio_xgb_model.json
├── portfolio_scaler.pkl
│
├── alpaca_config.json
├── .gitignore
│
└── README.md
```

---

# ⚠️ Disclaimer

This software is for educational and research purposes only.

Algorithmic trading involves substantial financial risk. Past performance and backtests do not guarantee future profitability.
