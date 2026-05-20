# 🤖 Institutional Cross-Sectional Alpha Predictor & Balanced Trading Engine

A quantitative, bi-directional algorithmic trading system built to isolate cross-sectional market alpha, eliminate historical target leakage (look-ahead bias), and execute risk-balanced long/short portfolio rotations via the Alpaca API.

---

## 📊 Core System Architecture

The pipeline is split into isolated, sequential modules designed to mimic institutional quant workflows:

1. **Universe Filtering (`get_universe.py`)**: Gathers the liquid asset universe.
2. **Feature Engineering (`feature_engine.py`)**: Computes macro indicators, technical features, and cross-sectional ranks with a strict 1-day lag to preserve timeline purity.
3. **Model Training (`train_model.py`)**: Trains an XGBoost Classifier on historical data using a chronological (non-random) split to prevent backtest leakage.
4. **Live Evaluation (`live_portfolio.py`)**: Extracts the current market state, scales features, and filters out low-confidence assets using a strict 54% hybrid probability threshold.
5. **Execution Router (`execute_trades.py`)**: Splits risk capital into independent 50% Long and 50% Short allocation buckets to maintain a balanced, directional-neutral profile.
6. **Automation Layer (`orchestrate_cron.py` & `live_desk.py`)**: Background daemons tracking high-precision clock intervals to handle hands-free live execution.

---

## 🛠️ Step-by-Step Installation & Setup

### 1. Prerequisites
Ensure your machine has Python 3.13+ installed. Verify your local installation by running:
```powershell
python --version
