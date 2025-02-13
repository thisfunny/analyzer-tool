import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


#########################################
# Helper to extract a scalar timestamp and convert it
#########################################
def get_timestamp(val):
    """Ensure val is a scalar timestamp and convert it to np.datetime64."""
    if isinstance(val, pd.Series):
        # If val is a Series (with one element), extract that element.
        val = val.iloc[0]
    return pd.Timestamp(val).to_datetime64()


#########################################
# Define Divergence Detection Function
#########################################
def check_divergence(price_pivots, rsi_pivots, pivot_type):
    """
    Checks for divergences between price and RSI pivots.

    For pivot_type 'low': bullish divergence (price makes a lower low while RSI makes a higher low).
    For pivot_type 'high': bearish divergence (price makes a higher high while RSI makes a lower high).

    Returns a list of timestamps where divergences are detected.
    """
    divergences = []
    # Loop through consecutive price pivot pairs.
    for i in range(1, len(price_pivots)):
        # Use the helper to get scalar timestamps as NumPy datetime64.
        lower_bound = get_timestamp(price_pivots.iloc[i - 1]['timestamp'])
        upper_bound = get_timestamp(price_pivots.iloc[i]['timestamp'])

        # Use the underlying NumPy array for the timestamp comparisons.
        ts_array = rsi_pivots['timestamp'].values

        # Select RSI pivots strictly between the two price pivot timestamps.
        rsi_in_window = rsi_pivots[(ts_array > lower_bound) & (ts_array < upper_bound)]

        # Build the extended RSI DataFrame.
        extended_rsi_df = pd.DataFrame()

        # 1. Get the last RSI pivot at or before the lower bound.
        rsi_before = rsi_pivots[ts_array <= lower_bound]
        if not rsi_before.empty:
            pivot_before = rsi_before.iloc[-1]
            extended_rsi_df = pd.concat([extended_rsi_df, pd.DataFrame([pivot_before])], ignore_index=True)

        # 2. Include all RSI pivots within the window.
        if not rsi_in_window.empty:
            extended_rsi_df = pd.concat([extended_rsi_df, rsi_in_window], ignore_index=True)

        # 3. Get the first RSI pivot at or after the upper bound.
        rsi_after = rsi_pivots[ts_array >= upper_bound]
        if not rsi_after.empty:
            pivot_after = rsi_after.iloc[0]
            extended_rsi_df = pd.concat([extended_rsi_df, pd.DataFrame([pivot_after])], ignore_index=True)

        # Skip this interval if no RSI data is available.
        if extended_rsi_df.empty:
            continue

        # Use the first and last rows of the extended RSI set for divergence comparison.
        first_rsi = extended_rsi_df.iloc[0]
        last_rsi = extended_rsi_df.iloc[-1]

        # Now, explicitly cast the pivot values to float before comparing.
        if pivot_type == 'low':
            try:
                low_current = float(price_pivots.iloc[i]['low_pivot'])
                low_prev = float(price_pivots.iloc[i - 1]['low_pivot'])
                rsi_low_first = float(first_rsi['rsi_low_pivot'])
                rsi_low_last = float(last_rsi['rsi_low_pivot'])
            except Exception as e:
                continue  # Skip if conversion fails.
            # Bullish divergence: price makes a lower low while RSI makes a higher low.
            if low_current < low_prev and rsi_low_last > rsi_low_first:
                # Extract the scalar timestamp value.
                divergences.append(price_pivots.iloc[i]['timestamp'].item())  # Use .item() to get the scalar value
        elif pivot_type == 'high':
            try:
                high_current = float(price_pivots.iloc[i]['high_pivot'])
                high_prev = float(price_pivots.iloc[i - 1]['high_pivot'])
                rsi_high_first = float(first_rsi['rsi_high_pivot'])
                rsi_high_last = float(last_rsi['rsi_high_pivot'])
            except Exception as e:
                continue  # Skip if conversion fails.
            # Bearish divergence: price makes a higher high while RSI makes a lower high.
            if high_current > high_prev and rsi_high_last < rsi_high_first:
                # Extract the scalar timestamp value.
                divergences.append(price_pivots.iloc[i]['timestamp'].item())  # Use .item() to get the scalar value

    return divergences


#########################################
# PART 1: Data Fetching, Processing, and Plotting
#########################################

# Fetch BTC/USD historical data.
symbol = "BTC-USD"
df = yf.download(symbol, interval="1d", period="1y")  # daily data for one year

# Reset index if Datetime is in the index.
if isinstance(df.index, pd.DatetimeIndex):
    df = df.reset_index()

# Rename columns to lowercase for consistency.
df.rename(columns={"Date": "timestamp", "Datetime": "timestamp", "High": "high", "Low": "low", "Close": "close"},
          inplace=True)

# Ensure a 'timestamp' column exists and convert it to datetime.
if "timestamp" not in df.columns:
    df["timestamp"] = df.index
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Calculate RSI.
window = 14
delta = df['close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
rs = gain / loss
df['rsi'] = 100 - (100 / (1 + rs))

# Find pivot points using a defined neighboring order (order=4).
order = 4  # Number of neighboring points to compare.
df['high_pivot'] = np.nan
df['low_pivot'] = np.nan
df['rsi_high_pivot'] = np.nan
df['rsi_low_pivot'] = np.nan

high_idx = argrelextrema(df['high'].values, np.greater, order=order)[0]
low_idx = argrelextrema(df['low'].values, np.less, order=order)[0]
rsi_high_idx = argrelextrema(df['rsi'].values, np.greater, order=order)[0]
rsi_low_idx = argrelextrema(df['rsi'].values, np.less, order=order)[0]

df.loc[df.index[high_idx], 'high_pivot'] = df.iloc[high_idx]['high'].values
df.loc[df.index[low_idx], 'low_pivot'] = df.iloc[low_idx]['low'].values
df.loc[df.index[rsi_high_idx], 'rsi_high_pivot'] = df.iloc[rsi_high_idx]['rsi'].values
df.loc[df.index[rsi_low_idx], 'rsi_low_pivot'] = df.iloc[rsi_low_idx]['rsi'].values

# Save pivot data to CSV (optional).
pivot_data = df[['timestamp', 'high_pivot', 'low_pivot', 'rsi_high_pivot', 'rsi_low_pivot']].dropna(how='all')
csv_filename = "btc_pivots.csv"
pivot_data.to_csv(csv_filename, index=False)
print(f"Pivots saved to {csv_filename}")

# Define RSI levels to monitor for signals.
rsi_levels = [12, 14, 31, 35, 49, 51, 64, 68, 86, 88]

# Initialize signal columns with None (object dtype) to avoid dtype warnings.
df['signal'] = None  # Will be 'buy' or 'sell'.
df['signal_price'] = np.nan  # Price at which the signal occurs.

# Detect signals based on RSI crossing defined levels.
for i in range(1, len(df)):
    for level in rsi_levels:
        # Buy signal: RSI crosses level from above.
        if df['rsi'].iloc[i - 1] > level and df['rsi'].iloc[i] <= level:
            df.at[df.index[i], 'signal'] = 'buy'
        # Sell signal: RSI crosses level from below.
        elif df['rsi'].iloc[i - 1] < level and df['rsi'].iloc[i] >= level:
            df.at[df.index[i], 'signal'] = 'sell'

# Save signals to CSV (optional).
signals = df[['timestamp', 'signal', 'close']].dropna()
signals_filename = "btc_signals.csv"
signals.to_csv(signals_filename, index=False)
print(f"Signals saved to {signals_filename}")

#########################################
# Prepare Pivot Data for Divergence Detection
#########################################
# Ensure timestamps are datetime.
pivot_data['timestamp'] = pd.to_datetime(pivot_data['timestamp'])

# Extract price and RSI pivot data for lows and highs.
price_low_pivots = pivot_data[pivot_data['low_pivot'].notna()][['timestamp', 'low_pivot']]
price_high_pivots = pivot_data[pivot_data['high_pivot'].notna()][['timestamp', 'high_pivot']]
rsi_low_pivots = pivot_data[pivot_data['rsi_low_pivot'].notna()][['timestamp', 'rsi_low_pivot']]
rsi_high_pivots = pivot_data[pivot_data['rsi_high_pivot'].notna()][['timestamp', 'rsi_high_pivot']]

# Sort and reset the indices so that the comparisons in check_divergence work properly.
price_low_pivots = price_low_pivots.sort_values('timestamp').reset_index(drop=True)
rsi_low_pivots = rsi_low_pivots.sort_values('timestamp').reset_index(drop=True)
price_high_pivots = price_high_pivots.sort_values('timestamp').reset_index(drop=True)
rsi_high_pivots = rsi_high_pivots.sort_values('timestamp').reset_index(drop=True)

# Detect divergences.
bullish_divergences = check_divergence(price_low_pivots, rsi_low_pivots, pivot_type='low')
bearish_divergences = check_divergence(price_high_pivots, rsi_high_pivots, pivot_type='high')

print("Bullish Divergences detected at timestamps:", bullish_divergences)
print("Bearish Divergences detected at timestamps:", bearish_divergences)

#########################################
# Create Figure and Subplots
#########################################
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                               sharex=True, gridspec_kw={'height_ratios': [3, 1]})

# --- Price Plot (Top) ---
ax1.plot(df['timestamp'], df['close'], label="BTC Price", color='black', linewidth=1)
ax1.scatter(df['timestamp'], df['high_pivot'], color='red', label="High Pivot", marker='v', alpha=0.8)
ax1.scatter(df['timestamp'], df['low_pivot'], color='green', label="Low Pivot", marker='^', alpha=0.8)

# Plot buy/sell signals.
buy_signals = df[df['signal'] == 'buy']
sell_signals = df[df['signal'] == 'sell']
ax1.scatter(buy_signals['timestamp'], buy_signals['close'], color='blue', label="Buy Signal", marker='o', alpha=0.5,
            s=15)
ax1.scatter(sell_signals['timestamp'], sell_signals['close'], color='red', label="Sell Signal", marker='o', alpha=0.5,
            s=15)


#########################################
# Add Divergence Markers to the Price Chart
#########################################
def get_close_price(ts):
    """Helper function to retrieve the close price for a given timestamp."""
    row = df[df['timestamp'] == ts]
    if not row.empty:
        return row['close'].values[0]
    return np.nan


# Convert divergence timestamps to datetime.
bullish_div_dates = pd.to_datetime(bullish_divergences)
bearish_div_dates = pd.to_datetime(bearish_divergences)

# Get corresponding close prices for each divergence timestamp.
bullish_prices = [get_close_price(ts) for ts in bullish_div_dates]
bearish_prices = [get_close_price(ts) for ts in bearish_div_dates]

# Plot divergence markers on the price chart.
ax1.scatter(bullish_div_dates, bullish_prices, marker='*', s=150, color='lime',
            label="Bullish Divergence", edgecolors='black')
ax1.scatter(bearish_div_dates, bearish_prices, marker='*', s=150, color='magenta',
            label="Bearish Divergence", edgecolors='black')

ax1.set_ylabel("BTC Price (USD)")
ax1.set_title("BTC/USD Price with Pivot Points, Signals, and Divergences")
ax1.legend()

# --- RSI Plot (Bottom) ---
ax2.plot(df['timestamp'], df['rsi'], label="RSI", color='blue', linestyle='dashed')
ax2.scatter(df['timestamp'], df['rsi_high_pivot'], color='purple', label="RSI High Pivot", marker='v', alpha=0.8)
ax2.scatter(df['timestamp'], df['rsi_low_pivot'], color='cyan', label="RSI Low Pivot", marker='^', alpha=0.8)

# Add horizontal lines for the RSI levels.
for level in rsi_levels:
    ax2.axhline(level, linestyle="--", color="red", alpha=0.5)

ax2.set_ylabel("RSI")
ax2.set_xlabel("Date")
ax2.legend()

# Format the x-axis.
ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
