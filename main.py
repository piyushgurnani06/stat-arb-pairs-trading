# ============================================================
# 1. Imports and setup
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

plt.rcParams["figure.figsize"] = (12, 6)


# ============================================================
# 2. Download price data
# ============================================================

ticker_1 = "V"
ticker_2 = "MA"

start_date = "2019-01-01"
end_date = "2025-12-31"

data = yf.download([ticker_1, ticker_2], start=start_date, end=end_date, auto_adjust=True)

# Use adjusted close-equivalent prices
prices = data["Close"].dropna()

print("First 5 rows:")
print(prices.head())
print("\nLast 5 rows:")
print(prices.tail())


# ============================================================
# 3. Plot normalised price series
# ============================================================

normalised_prices = prices / prices.iloc[0] * 100
normalised_prices.plot(title=f"Normalised Prices: {ticker_1} vs {ticker_2}")
plt.ylabel("Normalised Price (Base = 100)")
plt.show()


# ============================================================
# 4. Split into training and testing sets
# ============================================================

train = prices.loc["2019-01-01":"2023-12-31"].copy()
test = prices.loc["2024-01-01":"2025-12-31"].copy()

print(f"Training period: {train.index.min().date()} to {train.index.max().date()}")
print(f"Testing period:  {test.index.min().date()} to {test.index.max().date()}")


# ============================================================
# 5. Cointegration test on training data only
# ============================================================

score, pvalue, _ = coint(train[ticker_1], train[ticker_2])

print(f"Cointegration test p-value on training set: {pvalue:.6f}")


# ============================================================
# 6. Estimate hedge ratio on training data only
# ============================================================

X_train = sm.add_constant(train[ticker_2])
model = sm.OLS(train[ticker_1], X_train).fit()

hedge_ratio = model.params[ticker_2]
intercept = model.params["const"]

print(f"Intercept: {intercept:.6f}")
print(f"Hedge ratio: {hedge_ratio:.6f}")


# ============================================================
# 7. Construct spread using fixed training hedge ratio
# ============================================================

train_spread = train[ticker_1] - (intercept + hedge_ratio * train[ticker_2])
test_spread = test[ticker_1] - (intercept + hedge_ratio * test[ticker_2])

train_spread.plot(title="Training Spread")
plt.axhline(train_spread.mean(), linestyle="--")
plt.ylabel("Spread")
plt.show()

test_spread.plot(title="Testing Spread")
plt.axhline(test_spread.mean(), linestyle="--")
plt.ylabel("Spread")
plt.show()


# ============================================================
# 8. Create rolling z-score function
# ============================================================

def compute_zscore(spread, window=60):
    rolling_mean = spread.rolling(window).mean()
    rolling_std = spread.rolling(window).std()
    zscore = (spread - rolling_mean) / rolling_std
    return zscore


window = 60

train_zscore = compute_zscore(train_spread, window=window)
test_zscore = compute_zscore(test_spread, window=window)

train_zscore.plot(title=f"Training Z-Score (Window = {window})")
plt.axhline(2, linestyle="--")
plt.axhline(-2, linestyle="--")
plt.axhline(0, linestyle="--")
plt.show()

test_zscore.plot(title=f"Testing Z-Score (Window = {window})")
plt.axhline(2, linestyle="--")
plt.axhline(-2, linestyle="--")
plt.axhline(0, linestyle="--")
plt.show()


# ============================================================
# 9. Trading rule function
# ============================================================

def generate_positions(zscore, entry_threshold=2.0, exit_threshold=0.5):
    position = pd.Series(index=zscore.index, dtype=float)

    # Enter positions
    position[zscore > entry_threshold] = -1   # short spread
    position[zscore < -entry_threshold] = 1   # long spread

    # Exit positions
    position[abs(zscore) < exit_threshold] = 0

    # Carry forward the last known position
    position = position.ffill()

    # Any remaining NaN at the start becomes flat
    position = position.fillna(0)

    return position


train_position = generate_positions(train_zscore, entry_threshold=2.0, exit_threshold=0.5)
test_position = generate_positions(test_zscore, entry_threshold=2.0, exit_threshold=0.5)


# ============================================================
# 10. Backtest function 
# ============================================================

def backtest_strategy(price_data, position, hedge_ratio, intercept=0.0):
    results = pd.DataFrame(index=price_data.index)
    results["price_1"] = price_data.iloc[:, 0]
    results["price_2"] = price_data.iloc[:, 1]
    results["position"] = position

    # Spread based on fixed hedge ratio
    results["spread"] = results["price_1"] - (intercept + hedge_ratio * results["price_2"])

    # Daily spread change
    results["spread_change"] = results["spread"].diff()

    # Shift position by 1 day so today's trade uses yesterday's signal
    results["shifted_position"] = results["position"].shift(1)

    # Strategy return
    results["strategy_returns"] = results["shifted_position"] * results["spread_change"]

    # Fill initial NaN
    results["strategy_returns"] = results["strategy_returns"].fillna(0)

    # Cumulative PnL
    results["cumulative_pnl"] = results["strategy_returns"].cumsum()

    return results


train_results = backtest_strategy(train, train_position, hedge_ratio, intercept)
test_results = backtest_strategy(test, test_position, hedge_ratio, intercept)


# ============================================================
# 11. Plot cumulative PnL
# ============================================================

train_results["cumulative_pnl"].plot(title="Training Cumulative PnL")
plt.ylabel("PnL")
plt.show()

test_results["cumulative_pnl"].plot(title="Testing Cumulative PnL")
plt.ylabel("PnL")
plt.show()


# ============================================================
# 12. Print summary statistics
# ============================================================

def print_performance_stats(results, label):
    total_pnl = results["strategy_returns"].sum()
    mean_daily_return = results["strategy_returns"].mean()
    std_daily_return = results["strategy_returns"].std()

    sharpe_ratio = np.nan
    if std_daily_return != 0:
        sharpe_ratio = (mean_daily_return / std_daily_return) * np.sqrt(252)

    num_trades = results["position"].diff().abs().fillna(0)
    num_trades = (num_trades > 0).sum()

    print(f"\n{label} Performance")
    print("-" * 40)
    print(f"Total PnL: {total_pnl:.4f}")
    print(f"Mean daily return: {mean_daily_return:.6f}")
    print(f"Std dev of returns: {std_daily_return:.6f}")
    print(f"Annualised Sharpe ratio: {sharpe_ratio:.4f}")
    print(f"Number of position changes: {num_trades}")


print_performance_stats(train_results, "Training")
print_performance_stats(test_results, "Testing")


# ============================================================
# 13. Optional: Plot spread with entry thresholds visually
# ============================================================

fig, ax = plt.subplots()
test_zscore.plot(ax=ax, title="Testing Z-Score with Trading Thresholds")
ax.axhline(2, linestyle="--", color = "red", label="Short Entry")
ax.axhline(-2, linestyle="--", color = "green", label="Long Entry")
ax.axhline(0.5, linestyle=":")
ax.axhline(-0.5, linestyle=":")
ax.axhline(0, linestyle="--")
ax.legend()
plt.show()
