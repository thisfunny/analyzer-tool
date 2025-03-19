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
    All configurable parameters are passed via a configuration dictionary.
    """

    def __init__(self, config):
        self.symbol = config.get("symbol", "BTC-USD")
        self.interval = config.get("interval", "1d")
        self.period = config.get("period", "1y")
        self.rsi_levels = config.get("rsi_levels", [12, 14, 31, 35, 49, 51, 64, 68, 86, 88])
        self.pivot_order = config.get("pivot_order", 4)
        # Support for multiple RSI windows.
        self.rsi_windows = config.get("rsi_windows", [14])
        self.primary_rsi_window = config.get("primary_rsi_window", self.rsi_windows[0])

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
        Calculate multiple RSI indicators based on the specified windows.
        The primary RSI (as specified by primary_rsi_window) is stored in 'rsi'
        for further analysis.
        """
        delta = self.df['close'].diff()
        for window in self.rsi_windows:
            gain = delta.where(delta > 0, 0).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            # Store each RSI with a distinct column name, e.g. "rsi_14", "rsi_8", etc.
            self.df[f'rsi_{window}'] = rsi
        # Set the primary RSI column for use in pivot and signal detection.
        self.df['rsi'] = self.df[f'rsi_{self.primary_rsi_window}']

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
        (Uses the primary RSI in self.df['rsi'] for RSI pivots.)
        """
        self.df['high_pivot'] = np.nan
        self.df['low_pivot'] = np.nan
        self.df['rsi_high_pivot'] = np.nan
        self.df['rsi_low_pivot'] = np.nan

        # Detect pivot indices.
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
        for i in range(1, len(self.df) - 3):
            atr = self.df['atr'].values[i]
            neighbor = max(self.df["open"].values[i + 1], self.df["close"].values[i + 1])
            if not np.isnan(self.df["high_pivot"].values[i]):
                if self.df["high"].values[i] - self.df["close"].values[i] >= atr * 0.8:
                    continue
                elif self.df["high_pivot"].values[i] - min(self.df["open"].values[i + 1],
                                                           self.df["close"].values[i + 1]) >= atr * 0.8:
                    continue
                elif self.df["high_pivot"].values[i] - min(self.df["open"].values[i + 2],
                                                           self.df["close"].values[i + 2]) >= atr * 0.8:
                    continue
                elif self.df["high_pivot"].values[i] - min(self.df["open"].values[i + 3],
                                                           self.df["close"].values[i + 3]) < atr * 0.8:
                    self.df.at[self.df.index[i], "high_pivot"] = np.nan
                else:
                    continue

            if not np.isnan(self.df["low_pivot"].values[i]):
                if self.df["close"].values[i] - self.df["low"].values[i] >= atr * 0.8:
                    continue
                elif neighbor - self.df["low_pivot"].values[i] >= atr * 0.8:
                    continue
                elif max(self.df["open"].values[i + 2], self.df["close"].values[i + 2]) - self.df["low_pivot"].values[
                    i] >= atr * 0.8:
                    continue
                elif max(self.df["open"].values[i + 3], self.df["close"].values[i + 3]) - self.df["low_pivot"].values[
                    i] < atr * 0.8:
                    self.df.at[self.df.index[i], "low_pivot"] = np.nan
                else:
                    continue

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
    Extends Synthesis with methods for detecting signals, divergences, engulfing, pinbars,
    merging analyses, and plotting.
    """

    def __init__(self, config):
        super().__init__(config)
        self.bullish_divergences = []
        self.bearish_divergences = []
        self.multi_rsi_divergences = {}

    def detect_signals(self):
        """
        Detect buy and sell signals based on the primary RSI crossing predefined levels.
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
        Detect bullish and bearish divergences using pivot data (primary RSI).
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

    # --- NEW METHODS FOR MULTI-RSI DIVERGENCE (as before) ---
    def compute_rsi_pivots_for_window(self, window):
        """
        Compute RSI pivot points for a specific RSI window.
        The RSI column is assumed to be named 'rsi_{window}'.
        Returns a DataFrame with columns: timestamp, rsi_{window}_high_pivot, rsi_{window}_low_pivot.
        """
        col = f'rsi_{window}'
        self.df[f'{col}_high_pivot'] = np.nan
        self.df[f'{col}_low_pivot'] = np.nan
        rsi_series = self.df[col].values
        high_idx = argrelextrema(rsi_series, np.greater, order=self.pivot_order)[0]
        low_idx = argrelextrema(rsi_series, np.less, order=self.pivot_order)[0]
        self.df.loc[self.df.index[high_idx], f'{col}_high_pivot'] = self.df.iloc[high_idx][col].values
        self.df.loc[self.df.index[low_idx], f'{col}_low_pivot'] = self.df.iloc[low_idx][col].values
        rsi_pivot_df = self.df[['timestamp', f'{col}_high_pivot', f'{col}_low_pivot']].dropna(how='all')
        return rsi_pivot_df

    def detect_divergences_multi_rsi(self):
        """
        Detect divergences for each RSI window specified in self.rsi_windows.
        For each RSI window, compute its pivot points and compare them with price pivots.
        Results are stored in self.multi_rsi_divergences as a dictionary keyed by RSI window.
        """
        price_pivots = self.df[['timestamp', 'high_pivot', 'low_pivot']].dropna(how='all').copy()
        price_low_pivots = price_pivots[['timestamp', 'low_pivot']]
        price_high_pivots = price_pivots[['timestamp', 'high_pivot']]
        divergences = {}
        for window in self.rsi_windows:
            col = f'rsi_{window}'
            rsi_pivot_df = self.compute_rsi_pivots_for_window(window)
            # Rename columns for compatibility.
            rsi_low_pivots = rsi_pivot_df[['timestamp', f'{col}_low_pivot']].rename(
                columns={f'{col}_low_pivot': 'rsi_low_pivot'})
            rsi_high_pivots = rsi_pivot_df[['timestamp', f'{col}_high_pivot']].rename(
                columns={f'{col}_high_pivot': 'rsi_high_pivot'})
            div_low = self.check_divergence(price_low_pivots, rsi_low_pivots, pivot_type='low')
            div_high = self.check_divergence(price_high_pivots, rsi_high_pivots, pivot_type='high')
            divergences[window] = {'bullish': div_low, 'bearish': div_high}
        print("Multi-RSI Divergences detected:")
        for window, div in divergences.items():
            print(f"RSI window {window}: Bullish: {div['bullish']}, Bearish: {div['bearish']}")
        self.multi_rsi_divergences = divergences

    def detect_engulfing(self):
        """
        Detect bullish and bearish engulfing candlestick patterns.
        Bullish: Previous candle red, current candle green with current body engulfing previous body.
        Bearish: Previous candle green, current candle red with current body engulfing previous body.
        """
        self.df['engulfing'] = None
        engulfing_signals = []
        for i in range(1, len(self.df)):
            # Bullish engulfing
            if (self.df['open'].values[i - 1] > self.df['close'].values[i - 1]) and (
                    self.df['open'].values[i] < self.df['close'].values[i]):
                if (self.df['open'].values[i] < self.df['close'].values[i - 1]) and (
                        self.df['close'].values[i] > self.df['open'].values[i - 1]):
                    self.df.at[self.df.index[i], 'engulfing'] = 'bullish'
                    engulfing_signals.append(self.df['timestamp'].values[i])
            # Bearish engulfing
            if (self.df['open'].values[i - 1] < self.df['close'].values[i - 1]) and (
                    self.df['open'].values[i] > self.df['close'].values[i]):
                if (self.df['open'].values[i] > self.df['close'].values[i - 1]) and (
                        self.df['close'].values[i] < self.df['open'].values[i - 1]):
                    self.df.at[self.df.index[i], 'engulfing'] = 'bearish'
                    engulfing_signals.append(self.df['timestamp'].values[i])
        self.engulfing_signals = engulfing_signals
        print(engulfing_signals)

    def detect_pinbar(self):
        """
        Detect pinbar candlestick patterns.
        Bullish pinbar: Long lower wick relative to the body and a small upper wick.
        Bearish pinbar: Long upper wick relative to the body and a small lower wick.
        """
        self.df['pinbar'] = None
        pinbar_signals = []
        for i in range(len(self.df)):
            open_price = self.df["open"].values[i]
            close_price = self.df['close'].values[i]
            high = self.df['high'].values[i]
            low = self.df['low'].values[i]
            atr = self.df['atr'].values[i]
            body = abs(close_price - open_price)
            if body == 0:
                continue
            lower_wick = min(open_price, close_price) - low
            upper_wick = high - max(open_price, close_price)
            if atr * 0.8 < (high - low) > 1.2 * atr:
                if lower_wick > 3 * body and upper_wick < 0.5 * body:
                    self.df.at[self.df.index[i], 'pinbar'] = 'bullish'
                    pinbar_signals.append(self.df['timestamp'].values[i])
                elif upper_wick > 3 * body and lower_wick < 0.5 * body:
                    self.df.at[self.df.index[i], 'pinbar'] = 'bearish'
                    pinbar_signals.append(self.df['timestamp'].values[i])
            self.pinbar_signals = pinbar_signals

    def get_close_price(self, ts):
        """
        Retrieve the close price from the DataFrame for a given timestamp.
        """
        row = self.df[self.df['timestamp'] == ts]
        if not row.empty:
            return row['close'].values[0]
        return np.nan

    def save_whole_df_to_csv(self):
        """
        Save the entire analysis DataFrame to a CSV file.
        """
        self.df.to_csv("analysis_data.csv")
        print("Whole analysis data saved to analysis_data.csv")

    def merge_analysis(self, tolerance=3):
        """
        Merge signals from different analysis methods using the same style as other defs.
        For each row (starting at 1), this function checks:
          - If a pinbar signal is present and the primary RSI is within a given tolerance
            of any defined RSI level.
          - If a bullish divergence is detected (timestamp present in bullish_divergences),
            a valid low pivot exists, and the RSI is near one of the RSI levels.
          - If a bearish divergence is detected (timestamp present in bearish_divergences),
            a valid high pivot exists, and the RSI is near one of the RSI levels.
          - If an engulfing pattern is present and its timestamp is in the respective divergence set.
        When conditions are met, a merged signal is created in the 'merged_signal' column.
        """
        self.df['merged_signal'] = None
        bullish_div_set = set(self.bullish_divergences)
        bearish_div_set = set(self.bearish_divergences)

        for i in range(1, len(self.df)):
            signals = []
            ts = self.df['timestamp'].values[i]
            rsi_val = self.df['rsi'].values[i]

            # Check Pinbar + RSI Level
            if pd.notnull(self.df['pinbar'].values[i]):
                for level in self.rsi_levels:
                    if abs(rsi_val - level) <= tolerance:
                        if self.df['pinbar'].values[i] == 'bullish':
                            signals.append("merged_buy_pinbar_rsi")
                        elif self.df['pinbar'].values[i] == 'bearish':
                            signals.append("merged_sell_pinbar_rsi")
                        break  # Stop after first match

            # Check Divergence + RSI + Pivots
            if ts in bullish_div_set:
                if pd.notnull(self.df['low_pivot'].values[i]) and any(
                        abs(rsi_val - lvl) <= tolerance for lvl in self.rsi_levels):
                    signals.append("merged_buy_divergence_rsi")
            if ts in bearish_div_set:
                if pd.notnull(self.df['high_pivot'].values[i]) and any(
                        abs(rsi_val - lvl) <= tolerance for lvl in self.rsi_levels):
                    signals.append("merged_sell_divergence_rsi")

            # Check Engulfing + Divergence
            if pd.notnull(self.df['engulfing'].values[i]):
                if self.df['engulfing'].values[i] == 'bullish' and ts in bullish_div_set:
                    signals.append("merged_buy_engulfing_divergence")
                elif self.df['engulfing'].values[i] == 'bearish' and ts in bearish_div_set:
                    signals.append("merged_sell_engulfing_divergence")

            if signals:
                self.df.at[self.df.index[i], 'merged_signal'] = ", ".join(signals)

    def plot_results(self):
        """
        Plot the candlestick chart with pivot points, signals, divergence markers,
        engulfing and pinbar patterns, and merged signals.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                                       sharex=True,
                                       gridspec_kw={'height_ratios': [3, 1]})

        # Prepare candlestick data.
        ohlc = self.df[['timestamp', 'open', 'high', 'low', 'close']].copy()
        ohlc['timestamp'] = ohlc['timestamp'].apply(mdates.date2num)
        candlestick_ohlc(ax1, ohlc.values, width=0.6, colorup='green', colordown='red', alpha=0.8)

        # Overlay validated pivot points.
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

        # Plot engulfing patterns.
        engulfing_bull = self.df[self.df['engulfing'] == 'bullish']
        engulfing_bear = self.df[self.df['engulfing'] == 'bearish']
        ax1.scatter(mdates.date2num(engulfing_bull['timestamp']), engulfing_bull['close'],
                    marker='D', s=100, color='gold', label='Bullish Engulfing', edgecolors='black')
        ax1.scatter(mdates.date2num(engulfing_bear['timestamp']), engulfing_bear['close'],
                    marker='D', s=100, color='darkred', label='Bearish Engulfing', edgecolors='black')

        # Plot pinbar patterns.
        pinbar_bull = self.df[self.df['pinbar'] == 'bullish']
        pinbar_bear = self.df[self.df['pinbar'] == 'bearish']
        ax1.scatter(mdates.date2num(pinbar_bull['timestamp']), pinbar_bull['close'],
                    marker='P', s=100, color='limegreen', label='Bullish Pinbar', edgecolors='black')
        ax1.scatter(mdates.date2num(pinbar_bear['timestamp']), pinbar_bear['close'],
                    marker='P', s=100, color='purple', label='Bearish Pinbar', edgecolors='black')

        # Plot merged signals.
        merged = self.df[self.df['merged_signal'].notna()]
        if not merged.empty:
            ax1.scatter(mdates.date2num(merged['timestamp']), merged['close'],
                        marker='X', s=120, color='black', label='Merged Signal', edgecolors='yellow')

        ax1.set_ylabel("Price (USD)")
        ax1.set_title(f"{self.symbol} Price Chart with Pivots, Signals, & Patterns")
        ax1.legend()

        # Plot RSI and its pivots.
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
