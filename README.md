🤖 Institutional Cross-Sectional Alpha Predictor & Balanced Trading Engine

A quantitative, bi-directional algorithmic trading system built to isolate cross-sectional market alpha, eliminate historical target leakage (look-ahead bias), and execute risk-balanced long/short portfolio rotations via the Alpaca API.

📊 Core System Architecture

The pipeline is split into isolated, sequential modules designed to mimic institutional quantitative workflows:

Universe Filtering (get_universe.py)
Gathers the liquid asset universe.
Feature Engineering (feature_engine.py)
Computes macro indicators, technical features, and cross-sectional ranks with a strict 1-day lag to preserve timeline purity.
Model Training (train_model.py)
Trains an XGBoost Classifier on historical data using a chronological (non-random) split to prevent backtest leakage.
Live Evaluation (live_portfolio.py)
Extracts the current market state, scales features, and filters out low-confidence assets using a strict 54% hybrid probability threshold.
Execution Router (execute_trades.py)
Splits risk capital into independent 50% Long and 50% Short allocation buckets to maintain a balanced, directional-neutral profile.
Automation Layer (orchestrate_cron.py & live_desk.py)
Background daemons tracking high-precision clock intervals to handle hands-free live execution.
🛠️ Installation & Setup
1. Prerequisites

Ensure your machine has Python 3.13+ installed.

Verify your local installation:

python --version
2. Navigate to the Project Directory

Open your terminal or PowerShell and move into the project folder:

cd C:\Users\theko\Stock_AI_Project
3. Install Required Dependencies

Install the necessary data science, machine learning, and brokerage API libraries:

pip install pandas numpy xgboost scikit-learn alpaca-trade-api yfinance joblib requests
4. Configure API Credentials

Create a file named alpaca_config.json in the root project directory.

Paste your credentials into the following structure:

{
  "ALPACA_API_KEY": "YOUR_LIVE_OR_PAPER_API_KEY",
  "ALPACA_SECRET_KEY": "YOUR_LIVE_OR_PAPER_SECRET_KEY",
  "ALPACA_BASE_URL": "https://paper-api.alpaca.markets",
  "NEWS_API_KEY": "YOUR_OPTIONAL_NEWS_API_KEY"
}

Note:
alpaca_config.json should be included in your .gitignore file so your credentials are never pushed to GitHub.

Example .gitignore entry:

alpaca_config.json
📈 Initial Model Compilation Workflow

Before launching automated operations, you must generate the historical feature matrix and train the predictive model.

Step 1 — Generate Master Features

Run the multi-stock feature engine:

python feature_engine.py
Expected Output
Balanced target distribution (roughly 50.3% vs 49.7%)
Exported dataset:
stock_features_master.csv
Step 2 — Train the Quantitative Model

Execute the training pipeline:

python train_model.py
Expected Output
Out-of-sample test accuracy:
Typically between 50.5% and 54.5%
Exported model artifacts:
portfolio_xgb_model.json
portfolio_scaler.pkl

In cross-sectional finance, even a small statistical edge can be meaningful when applied consistently across diversified portfolios.

🚀 Live Production Execution

To run the automated trading desk, open two separate PowerShell windows and keep them active.

Terminal 1 — Launch the Clock Monitor Orchestrator

The orchestrator monitors the system clock and automatically runs the pipeline at scheduled intervals:

6:50 AM → Selection generation
7:15 AM → Broker order routing
python orchestrate_cron.py
Terminal 2 — Launch the Risk Monitor Desk

The live desk monitors open positions, manages protective logic, and handles orderly post-market hibernation loops.

python live_desk.py
🛡️ Risk Management & Failsafes
Dynamic Capital Sizing

Capital allocation is calculated symmetrically between long and short exposure.

Example:

If the model identifies:
18 Shorts
0 Longs

The broker router automatically:

Restricts exposure to the short-side pool only
Caps exposure at 50% total capital
Leaves the remaining 50% in cash as a directional hedge

This prevents catastrophic overexposure during broad market rallies.

Confidence Thresholding

Any asset with directional probability between:

46.0% and 54.0%

is classified as statistical noise and removed from the execution queue.

This prevents low-conviction trades from entering the portfolio.

Timeline Insulation

All indicators and macro metrics are computed using finalized historical closing bars.

The system never assumes knowledge of:

current intraday closes
future candles
unreleased market information

This preserves strict timeline integrity and eliminates look-ahead bias.

📂 Example Project Structure
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
⚠️ Disclaimer

This software is for educational and research purposes only.

Algorithmic trading involves substantial financial risk. Past performance, backtests, and statistical edge estimates do not guarantee future profitability.

Use paper trading environments before deploying real capital.
