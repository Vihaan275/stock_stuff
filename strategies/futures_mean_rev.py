def simple_mean_rev(df1,day=20,day1=150,day2=300):

    df = df1.copy()

    #day 1 = 300, day 2 = 150 -> best results 

    df['fast_ra'] = df['Close'].rolling(day1).mean().shift(1)
    df['slow_ra'] = df['Close'].rolling(day2).mean().shift(1)

    df['sma'] = df['Close'].rolling(day).mean().shift(1)
    df['std'] = df['Close'].rolling(day).std().shift(1)
    df['z-score'] = ((df['Close'].shift(1)-df['sma'])/df['std'])

    df['signal'] = 0

    df.loc[(df['z-score']>=2) & (df['z-score']<=3.1) & (df['fast_ra']>df['slow_ra']) & (df['KER']<0.8),'signal'] = 1
    df.loc[(df['z-score']<=-2) & (df['z-score']>=-3.1) & (df['fast_ra']<df['slow_ra']) &(df['KER']<0.8),'signal'] = -1


    return df

#removing KER increases return but sharpe is 2.25
#df['KER']<0.7, sharpe is 3