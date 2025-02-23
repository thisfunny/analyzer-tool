import numpy as np
import pandas as pd
import yfinance


def validate_price_pivots():
    # df=pd.read_excel("rsi_analysis.xlsx")
    df = yfinance.download("BTC-USD", start="2024-01-01", end="2025-01-01")
    """
    Validate price pivot points using the ATR.
    For each high pivot, this method checks if the difference between the pivot and the
    higher of its immediate neighbors is at least (atr_multiplier * ATR). For a low pivot,
    it checks if the difference between the lower of its immediate neighbors and the pivot is
    at least (atr_multiplier * ATR). If not, the pivot is invalidated (set to NaN).

    Parameters:
        atr_multiplier (float): The multiplier for the ATR threshold (default: 1.0).
    """
    # Loop over the DataFrame (skip first and last row).
    for i in range(1, len(df) - 1):
        # Validate high pivot.
        if not np.isnan(df["high_pivot"][i]):
            prev_high = df["high"][i-1]
            next_high = df["high"][i+1]
            local_diff = df["high_pivot"][i] - max(prev_high, next_high)
            df["Sag"] = df["high_pivot"]
            # if local_diff < 8 * df["atr"][i]:
            #     df["high_pivot"][i] = 0
            #     print("sag")
    print(df["high_pivot"])
        # Validate low pivot.
        # if not np.isnan(self.df.loc[self.df.index[i], 'low_pivot']):
        #     prev_low = self.df.loc[self.df.index[i - 1], 'low']
        #     next_low = self.df.loc[self.df.index[i + 1], 'low']
        #     local_diff = min(prev_low, next_low) - self.df.loc[self.df.index[i], 'low_pivot']
        #     if local_diff < atr_multiplier * self.df.loc[self.df.index[i], 'atr']:
        #         self.df.at[self.df.index[i], 'low_pivot'] = np.nan

    # Update pivot_data with validated pivots.
    # self.pivot_data = self.df[['timestamp', 'high_pivot', 'low_pivot', 'rsi_high_pivot', 'rsi_low_pivot']].dropna(
    #     how='all')

validate_price_pivots()