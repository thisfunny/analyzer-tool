from market_analysis_defs import Integration


def main():
    config = {
        "symbol": "BTC-USD",
        "interval": "1d",
        "period": "5y",
        "rsi_levels": [12, 14, 31, 35, 49, 51, 64, 68, 86, 88],
        "pivot_order": 4,
        # Specify multiple RSI windows.
        "rsi_windows": [14, 8, 20],
        # Set the primary RSI window (used for signals, pivots, etc.)
        "primary_rsi_window": 14
    }

    analyzer = Integration(config)
    analyzer.fetch_data()
    analyzer.compute_rsi()
    analyzer.compute_atr()  # Compute ATR for volatility measures.
    analyzer.compute_pivots()
    analyzer.validate_price_pivots(atr_multiplier=1.0)
    analyzer.save_pivots_to_csv("pivots.csv")
    analyzer.detect_signals()
    analyzer.save_signals_to_csv("signals.csv")
    analyzer.detect_divergences()
    analyzer.detect_engulfing()
    analyzer.detect_pinbar()
    analyzer.save_whole_df_to_csv()
    analyzer.plot_results()


if __name__ == '__main__':
    main()
