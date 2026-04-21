# Statistical Arbitrage Strategy (Pairs Trading)

This project implements a statistical arbitrage strategy using pairs trading on equities.

## Overview
The strategy trades the spread between two historically related assets using cointegration and mean reversion.

Assets used:
- Visa (V)
- Mastercard (MA)

## Methodology

1. Download price data using yfinance
2. Split data into training (2019–2023) and testing (2024–2025)
3. Perform cointegration test on training data
4. Estimate hedge ratio using OLS regression
5. Construct spread
6. Compute rolling z-score
7. Generate trading signals:
   - Enter when |z| > 2
   - Exit when |z| < 0.5
8. Backtest strategy using lagged positions

## Key Features
- Out-of-sample testing (no look-ahead bias)
- Fixed hedge ratio from training set
- Rolling z-score signal generation
- Performance metrics (PnL, Sharpe ratio, trade count)

## How to Run
pip install -r requirements.txt
python main.py
