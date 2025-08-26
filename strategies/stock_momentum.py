def momentum(datafram):

    #step 1: find monthly return of each stock - done
    #step 2: find the mean of the monthly returns excluding the most recent month
    #step 3: find variance, and std 
    #buy the stocks with the highest mean, and short the stocks with lowest mean

    df = datafram.copy()

    skip_period = 1
    time_period = 2# needs to be 12

    #step 2
    df['mean'] = (df.groupby('ticker')['returns'].shift(skip_period).rolling(time_period).mean())#take the previous value down, take the average of the last few vlaues using that as reference. 
    df['std'] = df.groupby('ticker')['returns'].shift(skip_period).rolling(time_period).std()
    df['risk_adjusted'] = df['mean']/df['std']
    df['cum_returns'] = ((df.groupby('ticker')['Close'].shift(skip_period))/(df.groupby('ticker')['Close'].shift(skip_period+time_period)))-1

    df['signal'] = 

    print(df.head(16))





    
    

    
    


