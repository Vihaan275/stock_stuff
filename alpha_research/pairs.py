import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import numpy as np
import statsmodels.api as sm


# Function to compute cumulative return
def find_cum_return(df):
    df['daily_returns'] = df['close'].pct_change()
    df['cum_returns'] = (1 + df['daily_returns']).cumprod() - 1

# Load data
gold = pd.read_csv('./data/gold_yahoo.csv')
silver = pd.read_csv('./data/silver_yahoo.csv')

def pairs(gold,silver,num=3,std_away=1.06,bounds=0.42,delay=0,window=20,strat='exit_identified_signal',want=False):
    # Normalize column names
    gold.columns = gold.columns.str.lower()
    silver.columns = silver.columns.str.lower()

    gold['cum_returns'] = find_cum_return(gold)
    silver['cum_returns'] = find_cum_return(silver)

    # Convert 'date' column to datetime and set as index
    gold['date'] = pd.to_datetime(gold['date'])
    silver['date'] = pd.to_datetime(silver['date'])
    
    gold.set_index('date', inplace=True)
    silver.set_index('date', inplace=True)

    # Merge into wide format
    wide_df = pd.DataFrame({
        'gold': gold['close'],
        'silver': silver['close']
    }).dropna()

    wide_df['ratio'] = wide_df['gold']/wide_df['silver']

    #creating co-integration z-scores for better relation:

    # OLS regression: silver = beta * gold + intercept
    
    
    betas = []
    residuals = []

    for i in range(window, len(wide_df)):
        y = wide_df['silver'].iloc[i-window:i]
        x = wide_df['gold'].iloc[i-window:i]
        x = sm.add_constant(x)
        
        model = sm.OLS(y, x).fit()
        beta = model.params
        gold_val = wide_df['gold'].iloc[i]
        prediction = model.predict([[1, gold_val]])

        residual = wide_df['silver'].iloc[i] - prediction[0]
        
        betas.append(beta)
        residuals.append(residual)

    # Padding beginning with NaNs
    wide_df['coint_residual'] = [np.nan]*window + residuals
    # Normalize residual (Z-score)
    wide_df['coint_z_score'] = (
        wide_df['coint_residual'] - wide_df['coint_residual'].rolling(num).mean()
    ) / wide_df['coint_residual'].rolling(num).std()

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(wide_df.index, wide_df['coint_z_score'], color='black')
    plt.title('Gold/Silver coint Price Ratio')
    plt.xlabel('Date')
    plt.ylabel('Normalized Price')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    wide_df.to_csv('data/co_integ.csv')

    # Create long format for plotting
    combined = wide_df.reset_index().melt(id_vars='date', var_name='Asset', value_name='Cumulative Return')

    #graphing normalised ratio 
    wide_df['rolling_mean'] = wide_df['ratio'].rolling(num).mean()#PARAMETER
    wide_df['std'] = wide_df['ratio'].rolling(num).std()
    wide_df['normalised_value'] = (wide_df['ratio'] - wide_df['rolling_mean'])/(wide_df['std'])

    plt.figure(figsize=(12,6))
    plt.plot(wide_df.index, wide_df['normalised_value'],color='black')
    plt.title('Gold/Silver normalised price ratio')
    plt.xlabel('Date')
    plt.ylabel('normalised price')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    #making raw normalised signal
    wide_df['raw_signal'] = 0
    wide_df.loc[wide_df['normalised_value']>=std_away,'raw_signal'] = -1
    wide_df.loc[wide_df['normalised_value']<=-std_away,'raw_signal'] = 1

    wide_df['raw_signal'] = wide_df['raw_signal'].shift(1)

    #making coint normalised signal
    wide_df['coint_raw_signal'] = 0

    #first one should be +1
    wide_df.loc[wide_df['coint_z_score']>=std_away,'coint_raw_signal'] = 1 #short silver, long gold
    wide_df.loc[wide_df['coint_z_score']<=-std_away,'coint_raw_signal'] = -1#long silver short gold

    wide_df['coint_raw_signal'] = wide_df['coint_raw_signal'].shift(1)

    #making a Volatility filter
    wide_df['spread'] = wide_df['gold']-wide_df['silver']
    wide_df['volatility'] = wide_df['spread'].rolling(window).std()

    #making it so that we stick to the position until we reach 0 z-score:
    position = 0  # Current position: 1 (long), -1 (short), 0 (flat)

    for i in range(1, len(wide_df)):
        z = wide_df['coint_z_score'].iloc[i]

        # Entry signals
        if position == 0:
            if z >= std_away:
                position = 1  # Short spread
            elif z <= -std_away:
                position = -1   # Long spread

        # Exit signal: mean reversion
        elif position == -1 and z <= -bounds:
            position = 0
        elif position == 1 and z >= bounds:
            position = 0

        wide_df.at[wide_df.index[i], 'exit_identified_signal'] = position

    delay+=1

    wide_df[strat]=wide_df['exit_identified_signal'].shift(delay)#avoid lookahead bias

    wide_df.to_csv('data/Gold_Silver_results.csv')

    return wide_df

    #constants: how many std away, what number of days for lookback period to find mean



def cointegration_method(gold=gold, silver=silver, num=3):
    gold = gold.set_index('date')
    silver = silver.set_index('date')
    # Merge on index (e.g., DatetimeIndex)
    aligned = gold[['close']].rename(columns={'close': 'gold'}).join(
        silver[['close']].rename(columns={'close': 'silver'}),
        how='inner'
    )

    # OLS regression: silver = beta * gold + intercept
    X = sm.add_constant(aligned['gold'])
    model = sm.OLS(aligned['silver'], X).fit()
    aligned['coint_residual'] = aligned['silver'] - model.predict(X)

    # Normalize residual (Z-score)
    aligned['coint_z_score'] = (
        aligned['coint_residual'] - aligned['coint_residual'].rolling(num).mean()
    ) / aligned['coint_residual'].rolling(num).std()

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(aligned.index, aligned['coint_z_score'], color='black')
    plt.title('Gold/Silver Normalized Price Ratio')
    plt.xlabel('Date')
    plt.ylabel('Normalized Price')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    aligned.to_csv('data/co_integ.csv')

    return aligned  # Return if you want to inspect later


if __name__=="__main__":
    cointegration_method()