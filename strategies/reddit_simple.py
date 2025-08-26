import pandas as pd
import numpy as np

def test_run(df):
    df = df.copy()

    # Compute rolling mean of (high - low) over last 25 days
    df['rolling_range'] = (df['High'] - df['Low']).rolling(window=25).mean().shift(1)

    # Compute IBS = (close - low) / (high - low), avoid div by zero
    df['ibs'] = (df['Close'] - df['Low']) / (df['High'] - df['Low']).replace(0, 1e-6).shift(1)

    df['lower_band'] = df['High'].rolling(window=10).max().shift(1) - 2.5 * df['rolling_range']
    df['upper_band'] = df['Low'].rolling(window=10).min().shift(1) + 2.5 * df['rolling_range']

    df['yesterday_high'] = df['High'].shift(1)
    df['yesterday_low'] = df['Low'].shift(1)

    df['yesterday_close'] = df['Close'].shift(1)

    # Generate entry signals: long when close < lower_band and ibs < 0.3
    df['long_entry'] = (df['yesterday_close'] < df['lower_band']) & (df['ibs'] < 0.3)
    df['short_entry'] = (df['yesterday_close'] > df['upper_band']) & (df['ibs'] > 0.8)


    # Initialize position column
    df['position'] = 0

    in_trade = False
    trade_type = None  # "long" or "short"

    for i in range(len(df)):
        if not in_trade:
            if df.iloc[i]['long_entry']:
                in_trade = True
                trade_type = "long"
                df.at[df.index[i], 'position'] = -1
            elif df.iloc[i]['short_entry']:
                in_trade = True
                trade_type = "short"
                df.at[df.index[i], 'position'] = 1
            else:
                df.at[df.index[i], 'position'] = 0
        else:
            if trade_type == "long":
                df.at[df.index[i], 'position'] = -1
                if df.iloc[i]['yesterday_close'] > df.iloc[i]['yesterday_high']:
                    in_trade = False
                    trade_type = None
                    df.at[df.index[i], 'position'] = 0
            elif trade_type == "short":
                df.at[df.index[i], 'position'] = 1
                if df.iloc[i]['yesterday_close'] < df.iloc[i]['yesterday_low']:
                    in_trade = False
                    trade_type = None
                    df.at[df.index[i], 'position'] = 0



    df['signal'] = df['position']
    df['trade_change'] = df['signal'] != df['signal'].shift(1)
    return df