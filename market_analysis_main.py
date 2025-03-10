from market_analysis_defs import Integration


def main():
    # Define all tunable parameters here.
    config = {
        "symbol": "BTC-USD",  # Change to any ticker symbol you want to analyze.
        "interval": "1d",
        "period": "1y",
        "rsi_window": 14,
        "pivot_order": 4,
        "rsi_levels": [30, 50, 70]  # Set desired RSI levels.
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
    analyzer.plot_results()


if __name__ == '__main__':
    main()
