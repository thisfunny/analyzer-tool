import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc


class MarketAnalyzer:
    """
    A class for fetching market data, computing technical indicators (RSI, ATR, and pivot points),
    validating pivot points using ATR, detecting signals and divergences, and plotting the results
    using candlestick bars.
    """
    def __init__(self, symbol="BTC-USD", interval="1d", period="1y",
                 rsi_window=14, pivot_order=4, rsi_levels=None):
        if rsi_levels is None:
            rsi_levels = [12, 14, 31, 35, 49, 51, 64, 68, 86, 88]
        self.symbol = symbol
        self.interval = interval
        self.period = period
        self.rsi_window = rsi_window
        self.pivot_order = pivot_order
        self.rsi_levels = rsi_levels

        self.df = None           # Main DataFrame with price and indicator data
        self.pivot_data = None   # DataFrame containing only pivot data
        self.bullish_divergences = []  # List of timestamps with bullish divergence
        self.bearish_divergences = []  # List of timestamps with bearish divergence

    @staticmethod
    def get_timestamp(val):
        """
        Ensure val is a scalar timestamp and convert it to np.datetime64.
        """
        if isinstance(val, pd.Series):
            val = val.iloc[0]
        return pd.Timestamp(val).to_datetime64()

    @staticmethod
    def get_close_price(df, ts):
        """
        Retrieve the close price from the DataFrame for a given timestamp.
        """
        row = df[df['timestamp'] == ts]
        if not row.empty:
            return row['close'].values[0]
        return np.nan

    def fetch_data(self):
        """
        Fetch historical market data using yfinance and prepare the DataFrame.
        """
        self.df = yf.download(self.symbol, interval=self.interval, period=self.period)
        if isinstance(self.df.index, pd.DatetimeIndex):
            self.df = self.df.reset_index()
        # Rename columns to a consistent lowercase naming scheme.
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
        Calculate the RSI indicator and add it as a column in the DataFrame.
        """
        delta = self.df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=self.rsi_window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_window).mean()
        rs = gain / loss
        self.df['rsi'] = 100 - (100 / (1 + rs))

    def compute_atr(self, period=14):
        """
        Calculate the Average True Range (ATR) indicator over a given period.
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
        Calculate pivot points for both price and RSI.
        The detected pivot values are assigned as scalars by converting
        the computed arrays to lists.
        """
        self.df['high_pivot'] = np.nan
        self.df['low_pivot'] = np.nan
        self.df['rsi_high_pivot'] = np.nan
        self.df['rsi_low_pivot'] = np.nan

        # Detect price pivots.
        high_idx = argrelextrema(self.df['high'].values, np.greater, order=self.pivot_order)[0]
        low_idx = argrelextrema(self.df['low'].values, np.less, order=self.pivot_order)[0]
        # Detect RSI pivots.
        rsi_high_idx = argrelextrema(self.df['rsi'].values, np.greater, order=self.pivot_order)[0]
        rsi_low_idx = argrelextrema(self.df['rsi'].values, np.less, order=self.pivot_order)[0]

        # Assign pivot values by converting the arrays to lists.
        self.df.loc[self.df.index[high_idx], 'high_pivot'] = self.df['high'].iloc[high_idx].tolist()
        self.df.loc[self.df.index[low_idx], 'low_pivot'] = self.df['low'].iloc[low_idx].tolist()
        self.df.loc[self.df.index[rsi_high_idx], 'rsi_high_pivot'] = self.df['rsi'].iloc[rsi_high_idx].tolist()
        self.df.loc[self.df.index[rsi_low_idx], 'rsi_low_pivot'] = self.df['rsi'].iloc[rsi_low_idx].tolist()

        # Force conversion to numeric types.
        self.df['high_pivot'] = pd.to_numeric(self.df['high_pivot'], errors='coerce')
        self.df['low_pivot'] = pd.to_numeric(self.df['low_pivot'], errors='coerce')
        self.df['rsi_high_pivot'] = pd.to_numeric(self.df['rsi_high_pivot'], errors='coerce')
        self.df['rsi_low_pivot'] = pd.to_numeric(self.df['rsi_low_pivot'], errors='coerce')

        self.pivot_data = self.df[['timestamp', 'high_pivot', 'low_pivot', 'rsi_high_pivot', 'rsi_low_pivot']].dropna(how='all')

    def validate_price_pivots(self, atr_multiplier=1.0):
        """
        Validate price pivot points using the ATR.
        For each high pivot, this method checks if the difference between the pivot and the
        higher of its immediate neighbors is at least (atr_multiplier * ATR). For a low pivot,
        it checks if the difference between the lower of its immediate neighbors and the pivot is
        at least (atr_multiplier * ATR). If not, the pivot is invalidated (set to NaN).
        """
        for i in range(1, len(self.df) - 1):
            row = self.df.iloc[i]  # Get the row as a Series with scalar values.
            # Validate high pivot.
            if pd.notna(row['high_pivot']):
                prev_high = self.df.iloc[i - 1]['high']
                next_high = self.df.iloc[i + 1]['high']
                local_diff = row['high_pivot'] - max(prev_high, next_high)
                if local_diff < atr_multiplier * row['atr']:
                    self.df.at[self.df.index[i], 'high_pivot'] = np.nan
            # Validate low pivot.
            if pd.notna(row['low_pivot']):
                prev_low = self.df.iloc[i - 1]['low']
                next_low = self.df.iloc[i + 1]['low']
                local_diff = min(prev_low, next_low) - row['low_pivot']
                if local_diff < atr_multiplier * row['atr']:
                    self.df.at[self.df.index[i], 'low_pivot'] = np.nan

        self.pivot_data = self.df[['timestamp', 'high_pivot', 'low_pivot', 'rsi_high_pivot', 'rsi_low_pivot']].dropna(how='all')

    def save_pivots_to_csv(self, filename="btc_pivots.csv"):
        """
        Save the pivot data to a CSV file.
        """
        if self.pivot_data is not None:
            self.pivot_data.to_csv(filename, index=False)
            print(f"Pivots saved to {filename}")

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

    def save_signals_to_csv(self, filename="btc_signals.csv"):
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
                extended_rsi_df = pd.concat([extended_rsi_df,
                                             pd.DataFrame([pivot_before])],
                                            ignore_index=True)
            if not rsi_in_window.empty:
                extended_rsi_df = pd.concat([extended_rsi_df, rsi_in_window],
                                            ignore_index=True)
            rsi_after = rsi_pivots[ts_array >= upper_bound]
            if not rsi_after.empty:
                pivot_after = rsi_after.iloc[0]
                extended_rsi_df = pd.concat([extended_rsi_df,
                                             pd.DataFrame([pivot_after])],
                                            ignore_index=True)
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
        Detect bullish and bearish divergences using the pivot data.
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

    def plot_results(self):
        """
        Create a plot with two subplots:
          - The top subplot shows the BTC price using candlestick bars (with shadows),
            overlaid with validated pivot points, buy/sell signals, and divergence markers.
          - The bottom subplot shows the RSI with its pivot points and horizontal lines at key RSI levels.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                                       sharex=True,
                                       gridspec_kw={'height_ratios': [3, 1]})

        ohlc = self.df[['timestamp', 'open', 'high', 'low', 'close']].copy()
        ohlc['timestamp'] = ohlc['timestamp'].apply(mdates.date2num)
        candlestick_ohlc(ax1, ohlc.values, width=0.6, colorup='green', colordown='red', alpha=0.8)

        ax1.scatter(mdates.date2num(self.df['timestamp']), self.df['high_pivot'],
                    color='red', label="High Pivot", marker='v', alpha=0.8)
        ax1.scatter(mdates.date2num(self.df['timestamp']), self.df['low_pivot'],
                    color='green', label="Low Pivot", marker='^', alpha=0.8)

        buy_signals = self.df[self.df['signal'] == 'buy']
        sell_signals = self.df[self.df['signal'] == 'sell']
        ax1.scatter(mdates.date2num(buy_signals['timestamp']), buy_signals['close'],
                    color='blue', label="Buy Signal", marker='o', alpha=0.5, s=15)
        ax1.scatter(mdates.date2num(sell_signals['timestamp']), sell_signals['close'],
                    color='red', label="Sell Signal", marker='o', alpha=0.5, s=15)

        bullish_div_dates = pd.to_datetime(self.bullish_divergences)
        bearish_div_dates = pd.to_datetime(self.bearish_divergences)
        bullish_prices = [self.get_close_price(self.df, ts) for ts in bullish_div_dates]
        bearish_prices = [self.get_close_price(self.df, ts) for ts in bearish_div_dates]
        ax1.scatter(mdates.date2num(bullish_div_dates), bullish_prices,
                    marker='*', s=150, color='lime', label="Bullish Divergence", edgecolors='black')
        ax1.scatter(mdates.date2num(bearish_div_dates), bearish_prices,
                    marker='*', s=150, color='magenta', label="Bearish Divergence", edgecolors='black')

        ax1.set_ylabel("BTC Price (USD)")
        ax1.set_title("BTC/USD Price with Candlestick Bars, Validated Pivot Points, Signals, and Divergences")
        ax1.legend()

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


def main():
    analyzer = MarketAnalyzer()
    analyzer.fetch_data()
    analyzer.compute_rsi()
    analyzer.compute_atr()  # Compute ATR for volatility-based validations.
    analyzer.compute_pivots()
    analyzer.validate_price_pivots(atr_multiplier=1.0)  # Validate pivots using ATR.
    analyzer.save_pivots_to_csv()
    analyzer.detect_signals()
    analyzer.save_signals_to_csv()
    analyzer.detect_divergences()
    analyzer.plot_results()


if __name__ == '__main__':
    main()
