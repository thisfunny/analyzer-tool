import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Download BTC/USD data from Yahoo Finance
data = yf.download('BTC-USD', period='1y', interval='1d')
close = data['Close']

# RSI Calculation
def rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Compute RSI for different periods
data['RSI_8'] = rsi(close, 8)
data['RSI_15'] = rsi(close, 15)
data['RSI_21'] = rsi(close, 21)

# Function to detect divergences
def detect_divergence(rsi_series, price_series, lbL=3, lbR=1):
    divergences = []

    for i in range(lbL, len(rsi_series) - lbR):
        if rsi_series.iloc[i] < rsi_series.iloc[i - lbL] and rsi_series.iloc[i] < rsi_series.iloc[
            i + lbR]:  # Bullish Divergence
            if price_series.iloc[i] > price_series.iloc[i - lbL]:  # Confirm Hidden Bullish Divergence
                divergences.append((i, "Hidden Bullish"))
            else:
                divergences.append((i, "Regular Bullish"))

        if rsi_series.iloc[i] > rsi_series.iloc[i - lbL] and rsi_series.iloc[i] > rsi_series.iloc[
            i + lbR]:  # Bearish Divergence
            if price_series.iloc[i] < price_series.iloc[i - lbL]:  # Confirm Hidden Bearish Divergence
                divergences.append((i, "Hidden Bearish"))
            else:
                divergences.append((i, "Regular Bearish"))

    return divergences


# Detect divergences for each RSI period
divergences_8 = detect_divergence(data['RSI_8'], close)
divergences_15 = detect_divergence(data['RSI_15'], close)
divergences_21 = detect_divergence(data['RSI_21'], close)

# Plot RSI and divergences
fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
periods = [8, 15, 21]
divergences = [divergences_8, divergences_15, divergences_21]

for i, (rsi_period, divs) in enumerate(zip(periods, divergences)):
    ax[i].plot(data.index, data[f'RSI_{rsi_period}'], label=f'RSI {rsi_period}', color='purple')
    ax[i].axhline(70, linestyle='--', color='red', alpha=0.5)
    ax[i].axhline(30, linestyle='--', color='green', alpha=0.5)
    for signal, idx, value in divs:
        color = 'green' if 'Bullish' in signal else 'red'
        ax[i].scatter(data.index[idx], value, color=color, label=signal)
    ax[i].legend()
    ax[i].set_title(f'RSI {rsi_period} Divergence')

plt.show()
