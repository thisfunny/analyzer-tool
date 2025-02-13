import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ---------------------------
# 1. Download Data & Preprocess
# ---------------------------
symbol = "BTC-USD"
df = yf.download(symbol, interval="1d", period="1y")  # daily data

if isinstance(df.index, pd.DatetimeIndex):
    df = df.reset_index()

df.rename(columns={"Date": "timestamp", "Datetime": "timestamp",
                   "High": "high", "Low": "low", "Close": "close"}, inplace=True)

if "timestamp" not in df.columns:
    df["timestamp"] = df.index

# ---------------------------
# 2. Calculate ATR (Average True Range)
# ---------------------------
atr_period = 14
df['prev_close'] = df['close'].shift(1)

comp1 = df['high'] - df['low']
comp2 = (df['high'] - df['prev_close']).abs()
comp3 = (df['low'] - df['prev_close']).abs()

df['tr'] = pd.concat([comp1, comp2, comp3], axis=1).max(axis=1)
df['atr'] = df['tr'].rolling(window=atr_period, min_periods=atr_period).mean()

# ---------------------------
# 3. Calculate RSI
# ---------------------------
rsi_period = 14
delta = df['close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
rs = gain / loss
df['rsi'] = 100 - (100 / (1 + rs))

# ---------------------------
# 4. Identify Raw Pivot Points
# ---------------------------
order = 4  # number of neighboring points to compare

df['high_pivot'] = np.nan
df['low_pivot'] = np.nan
df['rsi_high_pivot'] = np.nan
df['rsi_low_pivot'] = np.nan

# Raw price pivots
raw_high_idx = argrelextrema(df['high'].values, np.greater, order=order)[0]
raw_low_idx  = argrelextrema(df['low'].values,  np.less,    order=order)[0]

df.loc[df.index[raw_high_idx], 'high_pivot'] = df.iloc[raw_high_idx]['high'].values
df.loc[df.index[raw_low_idx],  'low_pivot']  = df.iloc[raw_low_idx]['low'].values

# (Optional) RSI pivots:
rsi_high_idx = argrelextrema(df['rsi'].values, np.greater, order=order)[0]
rsi_low_idx  = argrelextrema(df['rsi'].values,  np.less,    order=order)[0]
df.loc[df.index[rsi_high_idx], 'rsi_high_pivot'] = df.iloc[rsi_high_idx]['rsi'].values
df.loc[df.index[rsi_low_idx],  'rsi_low_pivot']  = df.iloc[rsi_low_idx]['rsi'].values

# ---------------------------
# 5. Validate Pivots Using ATR (Opposite-Reference Method)
# ---------------------------
# For each raw high pivot, we find the most recent raw low pivot (if any) that came before it.
# We then compute the swing as the difference between the high pivot's price and that low pivot's price.
# If the swing is at least validation_factor * ATR (using ATR from the high pivot's bar),
# then we validate that high pivot.
# Similarly for raw low pivots.
validation_factor = 0.1

# Convert raw pivot indices to lists of integers.
raw_high = list(raw_high_idx)
raw_low  = list(raw_low_idx)

validated_high_idx = []
validated_low_idx = []

# Validate high pivots:
for hi in raw_high:
    lows_before = [j for j in raw_low if j < hi]
    if lows_before:
        last_low = max(lows_before)
        swing = df['high'].iloc[hi] - df['low'].iloc[last_low]
        # Use the underlying NumPy array to get a scalar ATR value.
        atr_value = float(df['atr'].values[hi])
        if swing >= validation_factor * atr_value:
            validated_high_idx.append(hi)

# Validate low pivots:
for li in raw_low:
    highs_before = [j for j in raw_high if j < li]
    if highs_before:
        last_high = max(highs_before)
        swing = df['high'].iloc[last_high] - df['low'].iloc[li]
        atr_value = float(df['atr'].values[li])
        if swing >= validation_factor * atr_value:
            validated_low_idx.append(li)

# Mark validated pivots in new columns.
df['validated_high_pivot'] = np.nan
df['validated_low_pivot'] = np.nan

df.loc[validated_high_idx, 'validated_high_pivot'] = df.loc[validated_high_idx, 'high']
df.loc[validated_low_idx, 'validated_low_pivot'] = df.loc[validated_low_idx, 'low']

# ---------------------------
# 6. Save CSV & Plotting
# ---------------------------
# We drop rows where both validated pivot columns are NaN.
pivot_data = df[['timestamp', 'high_pivot', 'low_pivot',
                 'validated_high_pivot', 'validated_low_pivot',
                 'rsi_high_pivot', 'rsi_low_pivot']]
pivot_data = pivot_data.dropna(subset=['validated_high_pivot', 'validated_low_pivot'], how='all')

csv_filename = "btc_pivots_with_atr_validation.csv"
pivot_data.to_csv(csv_filename, index=False)
print(f"Pivots saved to {csv_filename}")

# Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                               gridspec_kw={'height_ratios': [3, 1]})

# Price plot with raw and validated pivots.
ax1.plot(df['timestamp'], df['close'], label="BTC Price", color='black', linewidth=1)
ax1.scatter(df['timestamp'], df['high_pivot'], color='red', label="Raw High Pivot", marker='v', alpha=0.6)
ax1.scatter(df['timestamp'], df['low_pivot'], color='green', label="Raw Low Pivot", marker='^', alpha=0.6)
ax1.scatter(df['timestamp'], df['validated_high_pivot'], color='magenta', label="Validated High Pivot", marker='v', s=100, edgecolors='k')
ax1.scatter(df['timestamp'], df['validated_low_pivot'], color='orange', label="Validated Low Pivot", marker='^', s=100, edgecolors='k')
ax1.set_ylabel("BTC Price (USD)")
ax1.set_title("BTC/USD Price with Raw and ATR-Validated Pivot Points")
ax1.legend()

# RSI plot.
ax2.plot(df['timestamp'], df['rsi'], label="RSI", color='blue', linestyle='dashed')
ax2.scatter(df['timestamp'], df['rsi_high_pivot'], color='purple', label="RSI High Pivot", marker='v', alpha=0.8)
ax2.scatter(df['timestamp'], df['rsi_low_pivot'], color='cyan', label="RSI Low Pivot", marker='^', alpha=0.8)
ax2.axhline(70, linestyle="--", color="red", alpha=0.5)
ax2.axhline(30, linestyle="--", color="green", alpha=0.5)
ax2.set_ylabel("RSI")
ax2.set_xlabel("Date")
ax2.legend()

ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
