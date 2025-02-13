import pandas as pd
import numpy as np
import yfinance as yf
import talib
import matplotlib.pyplot as plt

# Fetch Bitcoin daily price data
def fetch_bitcoin_data():
    ticker = "BTC-USD"
    data = yf.download(ticker, start="2020-01-01", end="2025-01-29", interval="1d")
    return data

# Calculate RSI for multiple periods
def calculate_multiple_rsi(data, periods=[8, 15, 21]):
    for period in periods:
        data[f'RSI_{period}'] = talib.RSI(data['Close'], timeperiod=period)
    return data

# Detect divergences for a specific RSI column
def detect_divergence(data, rsi_column):
    # Initialize divergence columns
    data['Price_High'] = data['Close'].rolling(window=5, center=True).max()
    data['Price_Low'] = data['Close'].rolling(window=5, center=True).min()
    data[f'{rsi_column}_High'] = data[rsi_column].rolling(window=5, center=True).max()
    data[f'{rsi_column}_Low'] = data[rsi_column].rolling(window=5, center=True).min()

    # Bullish divergence: Price makes lower lows, RSI makes higher lows
    data[f'Bullish_Divergence_{rsi_column}'] = (data['Price_Low'] < data['Price_Low'].shift(1)) & \
                                              (data[f'{rsi_column}_Low'] > data[f'{rsi_column}_Low'].shift(1))

    # Bearish divergence: Price makes higher highs, RSI makes lower highs
    data[f'Bearish_Divergence_{rsi_column}'] = (data['Price_High'] > data['Price_High'].shift(1)) & \
                                               (data[f'{rsi_column}_High'] < data[f'{rsi_column}_High'].shift(1))

    return data

# Combine divergences for all RSI periods
def combine_divergences(data, periods=[8, 15, 21]):
    # Initialize combined divergence columns
    data['Bullish_Divergence_All'] = True
    data['Bearish_Divergence_All'] = True

    # Check if all RSI periods show the same divergence
    for period in periods:
        data['Bullish_Divergence_All'] &= data[f'Bullish_Divergence_RSI_{period}']
        data['Bearish_Divergence_All'] &= data[f'Bearish_Divergence_RSI_{period}']

    return data

# Plot price and RSI with combined divergences
def plot_combined_chart(data):
    plt.figure(figsize=(14, 10))

    # Plot Price
    plt.subplot(2, 1, 1)
    plt.plot(data['Close'], label='Price', color='blue')
    plt.scatter(data[data['Bullish_Divergence_All']].index,
                data[data['Bullish_Divergence_All']]['Close'],
                color='green', label='Bullish Divergence (All RSI)', marker='^', s=100)
    plt.scatter(data[data['Bearish_Divergence_All']].index,
                data[data['Bearish_Divergence_All']]['Close'],
                color='red', label='Bearish Divergence (All RSI)', marker='v', s=100)
    plt.title('Bitcoin Price with Combined Divergences (All RSI Periods)')
    plt.legend()

    # Plot RSI
    plt.subplot(2, 1, 2)
    for period in [8, 15, 21]:
        plt.plot(data[f'RSI_{period}'], label=f'RSI_{period}', alpha=0.7)
    plt.scatter(data[data['Bullish_Divergence_All']].index,
                data[data['Bullish_Divergence_All']]['RSI_21'],
                color='green', label='Bullish Divergence (All RSI)', marker='^', s=100)
    plt.scatter(data[data['Bearish_Divergence_All']].index,
                data[data['Bearish_Divergence_All']]['RSI_21'],
                color='red', label='Bearish Divergence (All RSI)', marker='v', s=100)
    plt.axhline(70, color='red', linestyle='--', alpha=0.5)
    plt.axhline(30, color='green', linestyle='--', alpha=0.5)
    plt.title('RSI with Combined Divergences (All RSI Periods)')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Generate a table of combined divergences
def generate_combined_divergence_table(data):
    bullish_divergences = data[data['Bullish_Divergence_All']]
    bearish_divergences = data[data['Bearish_Divergence_All']]

    # Combine bullish and bearish divergences into one table
    combined_table = pd.concat([
        bullish_divergences[['Close', 'RSI_8', 'RSI_15', 'RSI_21']].assign(Divergence_Type='Bullish (All RSI)'),
        bearish_divergences[['Close', 'RSI_8', 'RSI_15', 'RSI_21']].assign(Divergence_Type='Bearish (All RSI)')
    ]).sort_index()

    return combined_table

# Main function
def main():
    # Fetch data
    data = fetch_bitcoin_data()

    # Calculate RSI for multiple periods
    data = calculate_multiple_rsi(data, periods=[8, 15, 21])

    # Detect divergences for each RSI period
    for period in [8, 15, 21]:
        data = detect_divergence(data, f'RSI_{period}')

    # Combine divergences for all RSI periods
    data = combine_divergences(data, periods=[8, 15, 21])

    # Generate combined divergence table
    combined_table = generate_combined_divergence_table(data)
    print("\nCombined Divergences (All RSI Periods):")
    print(combined_table)

    # Plot combined chart
    plot_combined_chart(data)

if __name__ == "__main__":
    main()