import pandas as pd
import yfinance as yf
import pandas_ta as ta
import matplotlib.pyplot as plt


# Fetch Bitcoin daily price data
def fetch_bitcoin_data():
    ticker = "sol-USD"
    data = yf.download(ticker, start="2023-01-01", end="2023-04-29", interval="1d")

    print("Columns in downloaded data:", data.columns)  # Debugging step

    if 'Close' not in data.columns:
        raise ValueError("Error: 'Close' column is missing. Check Yahoo Finance data structure.")

    return data


# Calculate RSI for multiple periods using pandas_ta
def calculate_multiple_rsi(data, periods=[8, 15, 21]):
    # Flatten MultiIndex if needed
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]  # Keep only the first level (Price type)

    print("Columns after flattening:", data.columns)  # Debugging step

    if 'Close' not in data.columns:
        raise ValueError("Error: 'Close' column is still missing after flattening!")

    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')  # Ensure numeric values

    for period in periods:
        data[f'RSI_{period}'] = ta.rsi(data['Close'], length=period)

    return data

# Detect divergences for a specific RSI column
def detect_divergence(data, rsi_column, window=7):
    data['Price_High'] = data['Close'].rolling(window=window, center=True).max()
    data['Price_Low'] = data['Close'].rolling(window=window, center=True).min()
    data[f'{rsi_column}_High'] = data[rsi_column].rolling(window=window, center=True).max()
    data[f'{rsi_column}_Low'] = data[rsi_column].rolling(window=window, center=True).min()

    data[f'Bullish_Divergence_{rsi_column}'] = (data['Price_Low'] < data['Price_Low'].shift(1)) & \
                                               (data[f'{rsi_column}_Low'] > data[f'{rsi_column}_Low'].shift(1))

    data[f'Bearish_Divergence_{rsi_column}'] = (data['Price_High'] > data['Price_High'].shift(1)) & \
                                               (data[f'{rsi_column}_High'] < data[f'{rsi_column}_High'].shift(1))

    return data


# Combine divergences for all RSI periods
def combine_divergences(data, periods=[8, 15, 21], min_agreement=2):
    data['Bullish_Divergence_All'] = False
    data['Bearish_Divergence_All'] = False

    bullish_count = sum(data[f'Bullish_Divergence_RSI_{period}'] for period in periods)
    bearish_count = sum(data[f'Bearish_Divergence_RSI_{period}'] for period in periods)

    data['Bullish_Divergence_All'] = bullish_count >= min_agreement
    data['Bearish_Divergence_All'] = bearish_count >= min_agreement

    return data


# Plot price and RSI with combined divergences
def plot_combined_chart(data):
    plt.figure(figsize=(14, 10))

    # Plot Price
    plt.subplot(2, 1, 1)
    plt.plot(data.index, data['Close'], label='Price', color='blue')
    plt.scatter(data[data['Bullish_Divergence_All']].index,
                data[data['Bullish_Divergence_All']]['Close'],
                color='green', label='Bullish Divergence', marker='^', s=100)
    plt.scatter(data[data['Bearish_Divergence_All']].index,
                data[data['Bearish_Divergence_All']]['Close'],
                color='red', label='Bearish Divergence', marker='v', s=100)
    plt.title('Bitcoin Price with Divergences')
    plt.legend()

    # Plot RSI
    plt.subplot(2, 1, 2)
    for period in [8, 15, 21]:
        plt.plot(data.index, data[f'RSI_{period}'], label=f'RSI {period}', alpha=0.7)
    plt.scatter(data[data['Bullish_Divergence_All']].index,
                data[data['Bullish_Divergence_All']]['RSI_21'],
                color='green', label='Bullish Divergence', marker='^', s=100)
    plt.scatter(data[data['Bearish_Divergence_All']].index,
                data[data['Bearish_Divergence_All']]['RSI_21'],
                color='red', label='Bearish Divergence', marker='v', s=100)
    plt.axhline(70, color='red', linestyle='--', alpha=0.5)
    plt.axhline(30, color='green', linestyle='--', alpha=0.5)
    plt.title('RSI with Divergences')
    plt.legend()

    plt.tight_layout()
    plt.show()


# Generate a table of combined divergences
def generate_combined_divergence_table(data):
    bullish_divergences = data[data['Bullish_Divergence_All']]
    bearish_divergences = data[data['Bearish_Divergence_All']]

    combined_table = pd.concat([
        bullish_divergences[['Close', 'RSI_8', 'RSI_15', 'RSI_21']].assign(Divergence_Type='Bullish'),
        bearish_divergences[['Close', 'RSI_8', 'RSI_15', 'RSI_21']].assign(Divergence_Type='Bearish')
    ]).sort_index()

    return combined_table


# Main function
def main():
    data = fetch_bitcoin_data()
    data = calculate_multiple_rsi(data, periods=[8, 15, 21])

    for period in [8, 15, 21]:
        data = detect_divergence(data, f'RSI_{period}', window=3)

    data = combine_divergences(data, periods=[8, 15, 21], min_agreement=2)

    combined_table = generate_combined_divergence_table(data)
    print("\nCombined Divergences:")
    print(combined_table)

    plot_combined_chart(data)
    data.to_csv('bitcoin_data_with_rsi.csv')


if __name__ == "__main__":
    main()
