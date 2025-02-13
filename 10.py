import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Fetch BTC/USD historical data
symbol = "BTC-USD"
df = yf.download(symbol, interval="1d", period="1y")  # Change to '1d' for daily data

# Reset index if Datetime is in index
if isinstance(df.index, pd.DatetimeIndex):
    df = df.reset_index()

# Rename columns to lowercase for consistency
df.rename(columns={"Date": "timestamp", "Datetime": "timestamp", "High": "high", "Low": "low", "Close": "close"}, inplace=True)

# Ensure 'timestamp' column exists
if "timestamp" not in df.columns:
    df["timestamp"] = df.index  # Use index if not available

# Calculate RSI
window = 14
delta = df['close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
rs = gain / loss
df['rsi'] = 100 - (100 / (1 + rs))

# Find pivot points
order = 4  # Defines the number of neighboring points to compare
df['high_pivot'] = np.nan
df['low_pivot'] = np.nan
df['rsi_high_pivot'] = np.nan
df['rsi_low_pivot'] = np.nan

# Get pivot indices
high_idx = argrelextrema(df['high'].values, np.greater, order=order)[0]
low_idx = argrelextrema(df['low'].values, np.less, order=order)[0]
rsi_high_idx = argrelextrema(df['rsi'].values, np.greater, order=order)[0]
rsi_low_idx = argrelextrema(df['rsi'].values, np.less, order=order)[0]

# Ensure proper length match by using .iloc
df.loc[df.index[high_idx], 'high_pivot'] = df.iloc[high_idx]['high'].values
df.loc[df.index[low_idx], 'low_pivot'] = df.iloc[low_idx]['low'].values
df.loc[df.index[rsi_high_idx], 'rsi_high_pivot'] = df.iloc[rsi_high_idx]['rsi'].values
df.loc[df.index[rsi_low_idx], 'rsi_low_pivot'] = df.iloc[rsi_low_idx]['rsi'].values

# Filter non-null pivots
pivot_data = df[['timestamp', 'high_pivot', 'low_pivot', 'rsi_high_pivot', 'rsi_low_pivot']].dropna(how='all')

# Save to CSV
csv_filename = "btc_pivots.csv"
pivot_data.to_csv(csv_filename, index=False)

print(f"Pivots saved to {csv_filename}")

# Define RSI levels to monitor
rsi_levels = [12, 14, 31, 35, 49, 51, 64, 68, 86, 88]

# Initialize signal columns
df['signal'] = np.nan  # 'buy' or 'sell'
df['signal_price'] = np.nan  # Price at which the signal occurs

# Detect signals
for i in range(1, len(df)):
    for level in rsi_levels:
        # Check for buy signal (RSI crosses level from above)
        if df['rsi'].iloc[i-1] > level and df['rsi'].iloc[i] <= level:
            df.at[df.index[i], 'signal'] = 'buy'
            # df.at[df.index[i], 'signal_price'] = df['close'].iloc[i]
        # Check for sell signal (RSI crosses level from below)
        elif df['rsi'].iloc[i-1] < level and df['rsi'].iloc[i] >= level:
            df.at[df.index[i], 'signal'] = 'sell'
            # df.at[df.index[i], 'signal_price'] = df['close'].iloc[i]


# Filter signals
# print(df)
signals = df[['timestamp', 'signal', 'close']].dropna()

# Save signals to CSV
signals_filename = "btc_signals.csv"
signals.to_csv(signals_filename, index=False)

print(f"Signals saved to {signals_filename}")

# Create figure and subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

# --- Price Plot (Top) ---
ax1.plot(df['timestamp'], df['close'], label="BTC Price", color='black', linewidth=1)
ax1.scatter(df['timestamp'], df['high_pivot'], color='red', label="High Pivot", marker='v', alpha=0.8)
ax1.scatter(df['timestamp'], df['low_pivot'], color='green', label="Low Pivot", marker='^', alpha=0.8)

# Plot buy/sell signals
buy_signals = df[df['signal'] == 'buy']
sell_signals = df[df['signal'] == 'sell']
ax1.scatter(buy_signals['timestamp'], buy_signals['close'], color='blue', label="Buy Signal", marker='o', alpha=0.5, s=15)
ax1.scatter(sell_signals['timestamp'], sell_signals['close'], color='red', label="Sell Signal", marker='o', alpha=0.5, s=15)

ax1.set_ylabel("BTC Price (USD)")
ax1.set_title("BTC/USD Price with Pivot Points and Signals")
ax1.legend()

# --- RSI Plot (Bottom) ---
ax2.plot(df['timestamp'], df['rsi'], label="RSI", color='blue', linestyle='dashed')
ax2.scatter(df['timestamp'], df['rsi_high_pivot'], color='purple', label="RSI High Pivot", marker='v', alpha=0.8)
ax2.scatter(df['timestamp'], df['rsi_low_pivot'], color='cyan', label="RSI Low Pivot", marker='^', alpha=0.8)

# Add RSI levels
for level in rsi_levels:
    ax2.axhline(level, linestyle="--", color="red", alpha=0.5)

ax2.set_ylabel("RSI")
ax2.set_xlabel("Date")
ax2.legend()

# Format x-axis
ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45)

plt.tight_layout()  # Adjust layout
plt.show()