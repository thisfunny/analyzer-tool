import numpy as np
import pandas as pd
import yfinance as yf
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
import logging
from argparse import ArgumentParser

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def calculate_rsi(data, periods):
    """Calculate RSI for given periods."""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0))
    loss = (-delta.where(delta < 0, 0))

    for period in periods:
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        data[f'RSI_{period}'] = 100 - (100 / (1 + rs))
    return data


def find_extrema(data, order=10):
    """Find local maxima and minima in the data."""
    max_idx = argrelextrema(data.values, np.greater, order=order)[0]
    min_idx = argrelextrema(data.values, np.less, order=order)[0]
    return max_idx, min_idx


def detect_divergences(price, rsi, order=5):
    """Detect bullish and bearish divergences."""
    price_max_idx, price_min_idx = find_extrema(price, order=order)
    rsi_max_idx, rsi_min_idx = find_extrema(rsi, order=order)

    divergences = {'bullish': [], 'bearish': []}

    # Detect bullish divergences
    for i in range(1, min(len(price_min_idx), len(rsi_min_idx))):
        price_current = float(price.iloc[price_min_idx[i]])  # Ensure scalar extraction
        price_previous = float(price.iloc[price_min_idx[i - 1]])
        rsi_current = float(rsi.iloc[rsi_min_idx[i]])
        rsi_previous = float(rsi.iloc[rsi_min_idx[i - 1]])

        if price_current < price_previous and rsi_current > rsi_previous:
            divergences['bullish'].append((price_min_idx[i - 1], price_min_idx[i]))

    # Detect bearish divergences
    for i in range(1, min(len(price_max_idx), len(rsi_max_idx))):
        price_current = float(price.iloc[price_max_idx[i]])
        price_previous = float(price.iloc[price_max_idx[i - 1]])
        rsi_current = float(rsi.iloc[rsi_max_idx[i]])
        rsi_previous = float(rsi.iloc[rsi_max_idx[i - 1]])

        if price_current > price_previous and rsi_current < rsi_previous:
            divergences['bearish'].append((price_max_idx[i - 1], price_max_idx[i]))

    return divergences


def plot_data(data, divergences, selected_rsi, rsi_periods):
    """Plot price and RSI with divergences."""
    fig, axs = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    # Price chart
    axs[0].plot(data['Close'], label='Price', color='black')
    for div in divergences['bullish']:
        axs[0].plot(data.index[list(div)], data['Close'].iloc[list(div)], 'go-', label='Bullish Divergence')
    for div in divergences['bearish']:
        axs[0].plot(data.index[list(div)], data['Close'].iloc[list(div)], 'ro-', label='Bearish Divergence')
    axs[0].set_title('Price Chart with Divergences')
    axs[0].set_ylabel('Price')
    axs[0].grid()

    # RSI chart
    for period in rsi_periods:
        axs[1].plot(data[f'RSI_{period}'], label=f'RSI_{period}')
    axs[1].axhline(70, color='red', linestyle='--', alpha=0.7, label='Overbought (70)')
    axs[1].axhline(30, color='green', linestyle='--', alpha=0.7, label='Oversold (30)')
    axs[1].set_title('RSI Chart')
    axs[1].set_xlabel('Date')
    axs[1].set_ylabel('RSI Value')
    axs[1].grid()
    axs[1].legend()

    plt.tight_layout()
    plt.show()


def parse_args():
    """Parse command-line arguments."""
    parser = ArgumentParser(description="Detect RSI divergences in financial data.")
    parser.add_argument('--ticker', type=str, default='BTC-USD', help="Ticker symbol (default: BTC-USD)")
    parser.add_argument('--period', type=str, default='2y', help="Data period (default: 2y)")
    parser.add_argument('--interval', type=str, default='1d', help="Data interval (default: 1d)")
    parser.add_argument('--rsi_periods', nargs='+', type=int, default=[8, 15, 21],
                        help="RSI periods (default: 8 15 21)")
    return parser.parse_args()


def main():
    """Main function to execute the script."""
    args = parse_args()
    ticker = args.ticker
    period = args.period
    interval = args.interval
    rsi_periods = args.rsi_periods

    logging.info(f"Fetching data for ticker: {ticker}")
    try:
        data = yf.download(ticker, period=period, interval=interval)
        if data.empty:
            raise ValueError("No data fetched from Yahoo Finance.")
    except Exception as e:
        logging.error(f"Error fetching data: {e}")
        return

    data = calculate_rsi(data, rsi_periods)
    selected_rsi = f'RSI_{rsi_periods[1]}'  # Use the second RSI period for analysis
    divergences = detect_divergences(data['Close'], data[selected_rsi])

    plot_data(data, divergences, selected_rsi, rsi_periods)  # Pass rsi_periods here


if __name__ == "__main__":
    main()
