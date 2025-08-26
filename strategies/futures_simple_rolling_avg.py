def simple_rolling_avg(df):
    df['fast_ra'] = df['Close'].rolling(200).mean().shift(1)
    df['slow_ra'] = df['Close'].rolling(486).mean().shift(1)

    df['signal'] = 0  # Default: no position
    df.loc[(df['fast_ra'] > df['slow_ra']) & (df['KER']<0.3) & (df['ema_regression_slope']<0.3), 'signal'] = 1  # Long position
    df.loc[(df['fast_ra'] < df['slow_ra']) & (df['KER']<0.3), 'signal'] = 0  # no position

    



    return df  # Return modified DataFrame


#KER for previous 20 periods
    # df['fast_ra'] = df['Close'].rolling(200).mean()
    # df['slow_ra'] = df['Close'].rolling(486).mean()
#f['signal'] = 0  # Default: no position
    # df.loc[(df['fast_ra'] > df['slow_ra']) & (df['KER']<0.3), 'signal'] = 1  # Long position
    # df.loc[(df['fast_ra'] < df['slow_ra']) & (df['KER']<0.3), 'signal'] = 0 # no position
    