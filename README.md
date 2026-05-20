# 🤖 Institutional Cross-Sectional Alpha Predictor & Balanced Trading Engine

A quantitative, bi-directional algorithmic trading system built to isolate cross-sectional market alpha, eliminate historical target leakage (look-ahead bias), and execute risk-balanced long/short portfolio rotations via the Alpaca API.

```

# 🛠️ Installation & Setup

## 1. Prerequisites

Ensure your machine has Python 3.13+ installed, as well as VS Code. If it doesn't, visit python.org to install.

Verify your local installation by opening a new terminal (Terminal --> New Terminal) and running:

```powershell
python --version
```

---

## 2. Open Your Project Folder

Once the github is downloaded, drag the zipped file to C:/Users/USERNAME.

Extract the file.

Make sure that you are in the correct folder. To do so in VS Code, click File --> Open Folder --> AIStock (AIStock should be in C:/Users/YOURUSERNAME/AIStock).

---

## 3. Install Required Dependencies

Run this in your terminal:

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

## 5. Get your API Keys

Go to the Alpaca Sign Up page and enter your name, email, and password.
Verify your email address through the link sent by Alpaca, then secure your account by enabling Multi-Factor Authentication (MFA) on your dashboard.
Once logged into the Alpaca web dashboard, click the account selector menu (usually in the upper-left corner) and select the Paper Trading environment.
Locate the API Keys panel on the right sidebar of the dashboard. Click Generate New Keys.
Copy and save your paper trading API Key and Secret and replace them in your alpaca_config.json. These will be different from your live account keys.

Paper trading API key replaces "YOUR_LIVE_OR_PAPER_API_KEY"
Secret Key replaces "YOUR_LIVE_OR_PAPER_SECRET_KEY"

Go to newsapi.com/register.
Register for your api key using your email.
When you get the api key, replace it in the json file where it says "YOUR_OPTIONAL_NEWS_API_KEY"

---

# What scripts to run (*RUN THESE FIRST*).

## Step 1 — Generate Master Features

Type this in and hit enter:

```powershell
python feature_engine.py
```

### Expected Output
- Balanced target distribution (~50/50)
- `stock_features_master.csv`

---

## Step 2 — Train the Quantitative Model

Type this in and hit enter:

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

Type this into the terminal before 6:50 AM to gather stock data:

```powershell
python orchestrate_cron.py
```

Runs:
- 6:50 AM → Selection generation
- 7:15 AM → Broker order routing

---

## Terminal 2 — Launch the Risk Monitor Desk

As well as this with orchestrate_cron.py

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
