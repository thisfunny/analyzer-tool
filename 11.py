import pandas as pd


def check_divergence(price_pivots, rsi_pivots, pivot_type):
    divergences = []

    # It is assumed that both price_pivots and rsi_pivots are sorted by timestamp.
    for i in range(1, len(price_pivots)):
        # Define the interval boundaries from consecutive price pivots.
        lower_bound = price_pivots.iloc[i - 1]['timestamp']
        upper_bound = price_pivots.iloc[i]['timestamp']

        # RSI pivots strictly between the two price pivot timestamps.
        rsi_in_window = rsi_pivots[
            (rsi_pivots['timestamp'] > lower_bound) &
            (rsi_pivots['timestamp'] < upper_bound)
            ]

        # Build the extended RSI DataFrame.
        extended_rsi_df = pd.DataFrame()

        # 1. Get the last RSI pivot that occurs at or before the lower bound.
        rsi_before = rsi_pivots[rsi_pivots['timestamp'] <= lower_bound]
        if not rsi_before.empty:
            pivot_before = rsi_before.iloc[-1]
            extended_rsi_df = pd.concat([extended_rsi_df, pd.DataFrame([pivot_before])], ignore_index=True)

        # 2. Include all RSI pivots within the window.
        if not rsi_in_window.empty:
            extended_rsi_df = pd.concat([extended_rsi_df, rsi_in_window], ignore_index=True)

        # 3. Get the first RSI pivot that occurs at or after the upper bound.
        rsi_after = rsi_pivots[rsi_pivots['timestamp'] >= upper_bound]
        if not rsi_after.empty:
            pivot_after = rsi_after.iloc[0]
            extended_rsi_df = pd.concat([extended_rsi_df, pd.DataFrame([pivot_after])], ignore_index=True)

        # If no RSI data is available, skip this interval.
        if extended_rsi_df.empty:
            continue

        # Use the first and last rows of the extended RSI set for the divergence comparison.
        first_rsi = extended_rsi_df.iloc[0]
        last_rsi = extended_rsi_df.iloc[-1]

        # Check divergence conditions.
        if pivot_type == 'low':
            # Bullish divergence: Price makes a lower low while the RSI makes a higher low.
            if (price_pivots.iloc[i]['low_pivot'] < price_pivots.iloc[i - 1]['low_pivot'] and
                    last_rsi['rsi_low_pivot'] > first_rsi['rsi_low_pivot']):
                divergences.append(price_pivots.iloc[i]['timestamp'])
        elif pivot_type == 'high':
            # Bearish divergence: Price makes a higher high while the RSI makes a lower high.
            if (price_pivots.iloc[i]['high_pivot'] > price_pivots.iloc[i - 1]['high_pivot'] and
                    last_rsi['rsi_high_pivot'] < first_rsi['rsi_high_pivot']):
                divergences.append(price_pivots.iloc[i]['timestamp'])

    return divergences


# Example usage:
# Load the CSV file.
df = pd.read_csv('btc_pivots.csv')

# Extract the relevant pivot data.
price_low_pivots = df[df['low_pivot'].notna()][['timestamp', 'low_pivot']]
price_high_pivots = df[df['high_pivot'].notna()][['timestamp', 'high_pivot']]
rsi_low_pivots = df[df['rsi_low_pivot'].notna()][['timestamp', 'rsi_low_pivot']]
rsi_high_pivots = df[df['rsi_high_pivot'].notna()][['timestamp', 'rsi_high_pivot']]

# Check for divergences.
bullish_divergences = check_divergence(price_low_pivots, rsi_low_pivots, pivot_type='low')
bearish_divergences = check_divergence(price_high_pivots, rsi_high_pivots, pivot_type='high')

print("Bullish Divergences detected at timestamps:", bullish_divergences)
print("Bearish Divergences detected at timestamps:", bearish_divergences)
