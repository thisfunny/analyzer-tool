import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc


class Synthesis:
    """
    Contains methods for data fetching and computing technical indicators.
    All configurable parameters are passed in via a configuration dictionary.
    """

    def __init__(self, config):
        self.symbol = config.get("symbol", "BTC-USD")
        self.interval = config.get("interval", "1d")
        self.period = config.get("period", "1y")
        self.rsi_window = config.get("rsi_window", 14)
        self.pivot_order = config.get("pivot_order", 4)
        self.rsi_levels = config.get("rsi_levels", [12, 14, 31, 35, 49, 51, 64, 68, 86, 88])

        self.df = None  # Main DataFrame with price and indicator data.
        self.pivot_data = None  # DataFrame with pivot-related data.

    @staticmethod
    def get_timestamp(val):
        """
        Convert a value to a NumPy datetime64 timestamp.
        """
        if isinstance(val, pd.Series):
            val = val.iloc[0]
        return pd.Timestamp(val).to_datetime64()

    def fetch_data(self):
        """
        Fetch historical market data using yfinance.
        """
        self.df = yf.download(self.symbol, interval=self.interval, period=self.period)
        if isinstance(self.df.index, pd.DatetimeIndex):
            self.df = self.df.reset_index()
        self.df.rename(columns={
            "Date": "timestamp",
            "Datetime": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close"
        }, inplace=True)
        if "timestamp" not in self.df.columns:
            self.df["timestamp"] = self.df.index
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])

    def compute_rsi(self):
        """
        Calculate the RSI indicator and add it as a column.
        """
        delta = self.df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=self.rsi_window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_window).mean()
        rs = gain / loss
        self.df['rsi'] = 100 - (100 / (1 + rs))

    def compute_atr(self, period=14):
        """
        Calculate the Average True Range (ATR) indicator.
        """
        high = self.df['high']
        low = self.df['low']
        close = self.df['close']
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        self.df['atr'] = true_range.rolling(window=period).mean()
        return self.df['atr']

    def compute_pivots(self):
        """
        Compute price and RSI pivot points.
        """
        self.df['high_pivot'] = np.nan
        self.df['low_pivot'] = np.nan
        self.df['rsi_high_pivot'] = np.nan
        self.df['rsi_low_pivot'] = np.nan

        # Detect pivots.
        high_idx = argrelextrema(self.df['high'].values, np.greater, order=self.pivot_order)[0]
        low_idx = argrelextrema(self.df['low'].values, np.less, order=self.pivot_order)[0]
        rsi_high_idx = argrelextrema(self.df['rsi'].values, np.greater, order=self.pivot_order)[0]
        rsi_low_idx = argrelextrema(self.df['rsi'].values, np.less, order=self.pivot_order)[0]

        self.df.loc[self.df.index[high_idx], 'high_pivot'] = self.df.iloc[high_idx]['high'].values
        self.df.loc[self.df.index[low_idx], 'low_pivot'] = self.df.iloc[low_idx]['low'].values
        self.df.loc[self.df.index[rsi_high_idx], 'rsi_high_pivot'] = self.df.iloc[rsi_high_idx]['rsi'].values
        self.df.loc[self.df.index[rsi_low_idx], 'rsi_low_pivot'] = self.df.iloc[rsi_low_idx]['rsi'].values

        self.pivot_data = self.df[['timestamp', 'high_pivot', 'low_pivot', 'rsi_high_pivot', 'rsi_low_pivot']].dropna(
            how='all')

    def validate_price_pivots(self, atr_multiplier=1.0):
        """
        Validate price pivots using an ATR-based threshold.
        """
        for i in range(1, len(self.df) - 1):
            if not np.isnan(self.df["high_pivot"].values[i]):
                if self.df["high_pivot"].values[i] - self.df["low"].values[i + 1] < atr_multiplier * \
                        self.df["atr"].values[i]:
                    self.df.at[self.df.index[i], "high_pivot"] = np.nan
                elif self.df["high_pivot"].values[i] - self.df["low"].values[i + 2] < atr_multiplier * \
                        self.df["atr"].values[i]:
                    self.df.at[self.df.index[i], "high_pivot"] = np.nan
                elif self.df["high_pivot"].values[i] - self.df["low"].values[i + 3] < atr_multiplier * \
                        self.df["atr"].values[i]:
                    self.df.at[self.df.index[i], "high_pivot"] = np.nan

            if not np.isnan(self.df["low_pivot"].values[i]):
                neighbor = min(self.df["low"].values[i - 1], self.df["low"].values[i + 1])
                if neighbor - self.df["low_pivot"].values[i] < atr_multiplier * self.df["atr"].values[i]:
                    self.df.at[self.df.index[i], "low_pivot"] = np.nan

        self.pivot_data = self.df[['timestamp', 'high_pivot', 'low_pivot', 'rsi_high_pivot', 'rsi_low_pivot']].dropna(
            how='all')

    def save_pivots_to_csv(self, filename="pivots.csv"):
        """
        Save pivot data to a CSV file.
        """
        if self.pivot_data is not None:
            self.pivot_data.to_csv(filename, index=False)
            print(f"Pivots saved to {filename}")


class Integration(Synthesis):
    """
    Extends Synthesis with methods for detecting signals, divergences, and plotting.
    """

    def __init__(self, config):
        super().__init__(config)
        self.bullish_divergences = []
        self.bearish_divergences = []

    def detect_signals(self):
        """
        Detect buy and sell signals based on RSI crossing predefined levels.
        """
        self.df['signal'] = None
        self.df['signal_price'] = np.nan

        for i in range(1, len(self.df)):
            for level in self.rsi_levels:
                if self.df['rsi'].iloc[i - 1] > level and self.df['rsi'].iloc[i] <= level:
                    self.df.at[self.df.index[i], 'signal'] = 'buy'
                elif self.df['rsi'].iloc[i - 1] < level and self.df['rsi'].iloc[i] >= level:
                    self.df.at[self.df.index[i], 'signal'] = 'sell'

    def save_signals_to_csv(self, filename="signals.csv"):
        """
        Save detected signals to a CSV file.
        """
        signals = self.df[['timestamp', 'signal', 'close']].dropna()
        signals.to_csv(filename, index=False)
        print(f"Signals saved to {filename}")

    def check_divergence(self, price_pivots, rsi_pivots, pivot_type):
        """
        Check for divergences between price and RSI pivot points.
        """
        divergences = []
        for i in range(1, len(price_pivots)):
            lower_bound = self.get_timestamp(price_pivots.iloc[i - 1]['timestamp'])
            upper_bound = self.get_timestamp(price_pivots.iloc[i]['timestamp'])
            ts_array = rsi_pivots['timestamp'].values

            rsi_in_window = rsi_pivots[(ts_array > lower_bound) & (ts_array < upper_bound)]
            extended_rsi_df = pd.DataFrame()

            rsi_before = rsi_pivots[ts_array <= lower_bound]
            if not rsi_before.empty:
                pivot_before = rsi_before.iloc[-1]
                extended_rsi_df = pd.concat([extended_rsi_df, pd.DataFrame([pivot_before])], ignore_index=True)

            if not rsi_in_window.empty:
                extended_rsi_df = pd.concat([extended_rsi_df, rsi_in_window], ignore_index=True)

            rsi_after = rsi_pivots[ts_array >= upper_bound]
            if not rsi_after.empty:
                pivot_after = rsi_after.iloc[0]
                extended_rsi_df = pd.concat([extended_rsi_df, pd.DataFrame([pivot_after])], ignore_index=True)

            if extended_rsi_df.empty:
                continue

            first_rsi = extended_rsi_df.iloc[0]
            last_rsi = extended_rsi_df.iloc[-1]

            if pivot_type == 'low':
                try:
                    low_current = float(price_pivots.iloc[i]['low_pivot'])
                    low_prev = float(price_pivots.iloc[i - 1]['low_pivot'])
                    rsi_low_first = float(first_rsi['rsi_low_pivot'])
                    rsi_low_last = float(last_rsi['rsi_low_pivot'])
                except Exception:
                    continue
                if low_current < low_prev and rsi_low_last > rsi_low_first:
                    divergences.append(price_pivots.iloc[i]['timestamp'].item())
            elif pivot_type == 'high':
                try:
                    high_current = float(price_pivots.iloc[i]['high_pivot'])
                    high_prev = float(price_pivots.iloc[i - 1]['high_pivot'])
                    rsi_high_first = float(first_rsi['rsi_high_pivot'])
                    rsi_high_last = float(last_rsi['rsi_high_pivot'])
                except Exception:
                    continue
                if high_current > high_prev and rsi_high_last < rsi_high_first:
                    divergences.append(price_pivots.iloc[i]['timestamp'].item())
        return divergences

    def detect_divergences(self):
        """
        Detect bullish and bearish divergences using pivot data.
        """
        self.pivot_data['timestamp'] = pd.to_datetime(self.pivot_data['timestamp'])

        price_low_pivots = self.pivot_data[self.pivot_data['low_pivot'].notna()][['timestamp', 'low_pivot']]
        price_high_pivots = self.pivot_data[self.pivot_data['high_pivot'].notna()][['timestamp', 'high_pivot']]
        rsi_low_pivots = self.pivot_data[self.pivot_data['rsi_low_pivot'].notna()][['timestamp', 'rsi_low_pivot']]
        rsi_high_pivots = self.pivot_data[self.pivot_data['rsi_high_pivot'].notna()][['timestamp', 'rsi_high_pivot']]

        price_low_pivots = price_low_pivots.sort_values('timestamp').reset_index(drop=True)
        rsi_low_pivots = rsi_low_pivots.sort_values('timestamp').reset_index(drop=True)
        price_high_pivots = price_high_pivots.sort_values('timestamp').reset_index(drop=True)
        rsi_high_pivots = rsi_high_pivots.sort_values('timestamp').reset_index(drop=True)

        self.bullish_divergences = self.check_divergence(price_low_pivots, rsi_low_pivots, pivot_type='low')
        self.bearish_divergences = self.check_divergence(price_high_pivots, rsi_high_pivots, pivot_type='high')

        print("Bullish Divergences detected at timestamps:", self.bullish_divergences)
        print("Bearish Divergences detected at timestamps:", self.bearish_divergences)

    def get_close_price(self, ts):
        """
        Retrieve the close price from the DataFrame for a given timestamp.
        """
        row = self.df[self.df['timestamp'] == ts]
        if not row.empty:
            return row['close'].values[0]
        return np.nan

    def plot_results(self):
        """
        Plot the candlestick chart with pivot points, signals, divergence markers, and the RSI indicator.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                                       sharex=True,
                                       gridspec_kw={'height_ratios': [3, 1]})

        # Prepare candlestick data.
        ohlc = self.df[['timestamp', 'open', 'high', 'low', 'close']].copy()
        ohlc['timestamp'] = ohlc['timestamp'].apply(mdates.date2num)
        candlestick_ohlc(ax1, ohlc.values, width=0.6, colorup='green', colordown='red', alpha=0.8)

        # Overlay pivot points.
        ax1.scatter(mdates.date2num(self.df['timestamp']), self.df['high_pivot'],
                    color='red', label="High Pivot", marker='v', alpha=0.8)
        ax1.scatter(mdates.date2num(self.df['timestamp']), self.df['low_pivot'],
                    color='green', label="Low Pivot", marker='^', alpha=0.8)

        # Plot buy/sell signals.
        buy_signals = self.df[self.df['signal'] == 'buy']
        sell_signals = self.df[self.df['signal'] == 'sell']
        ax1.scatter(mdates.date2num(buy_signals['timestamp']), buy_signals['close'],
                    color='blue', label="Buy Signal", marker='o', alpha=0.5, s=15)
        ax1.scatter(mdates.date2num(sell_signals['timestamp']), sell_signals['close'],
                    color='red', label="Sell Signal", marker='o', alpha=0.5, s=15)

        # Add divergence markers.
        bullish_div_dates = pd.to_datetime(self.bullish_divergences)
        bearish_div_dates = pd.to_datetime(self.bearish_divergences)
        bullish_prices = [self.get_close_price(ts) for ts in bullish_div_dates]
        bearish_prices = [self.get_close_price(ts) for ts in bearish_div_dates]
        ax1.scatter(mdates.date2num(bullish_div_dates), bullish_prices,
                    marker='*', s=150, color='lime', label="Bullish Divergence", edgecolors='black')
        ax1.scatter(mdates.date2num(bearish_div_dates), bearish_prices,
                    marker='*', s=150, color='magenta', label="Bearish Divergence", edgecolors='black')

        ax1.set_ylabel("Price (USD)")
        ax1.set_title(f"{self.symbol} Price Chart with Pivots, Signals, & Divergences")
        ax1.legend()

        # Plot RSI.
        ax2.plot(self.df['timestamp'], self.df['rsi'],
                 label="RSI", color='blue', linestyle='dashed')
        ax2.scatter(self.df['timestamp'], self.df['rsi_high_pivot'],
                    color='purple', label="RSI High Pivot", marker='v', alpha=0.8)
        ax2.scatter(self.df['timestamp'], self.df['rsi_low_pivot'],
                    color='cyan', label="RSI Low Pivot", marker='^', alpha=0.8)

        for level in self.rsi_levels:
            ax2.axhline(level, linestyle="--", color="red", alpha=0.5)
        ax2.set_ylabel("RSI")
        ax2.set_xlabel("Date")
        ax2.legend()

        ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
