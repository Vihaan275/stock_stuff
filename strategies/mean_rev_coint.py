import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import numpy as np

# Example: Load your two time series as pandas Series
# Replace this with your real data
# Y = pd.Series(...)  # dependent asset (e.g. stock A)
# X = pd.Series(...)  # independent asset (e.g. stock B)

def get_half_life(df1,df2):
    X = np.log(df1['Close'])
    Y = np.log(df2['Close'])


    # 1. Regress Y on X to get hedge ratio beta and calculate spread
    X_const = sm.add_constant(X)
    model = sm.OLS(Y, X_const).fit()
    beta = model.params[1]
    const = model.params[0]

    spread = Y - (beta * X + const)

    print(f"Hedge ratio (beta): {beta:.4f}")
    print(f"Intercept: {const:.4f}")

    # 2. Run Augmented Dickey-Fuller test on spread to check stationarity
    adf_result = adfuller(spread)
    print("\nADF Test on Spread:")
    print(f"ADF Statistic: {adf_result[0]:.4f}")
    print(f"p-value: {adf_result[1]:.4f}")
    for key, value in adf_result[4].items():
        print(f"Critical Value {key}: {value:.4f}")

    if adf_result[1] < 0.05:
        print("Spread is stationary. Proceed with mean reversion strategy.")
    else:
        print("Spread is NOT stationary. Be cautious.")

    # 3. Calculate half-life of mean reversion
    # Δspread = λ * spread_lag + ε

    spread_lag = spread.shift(1).dropna()
    delta_spread = spread.diff().dropna()

    spread_lag = spread_lag.loc[delta_spread.index]  # align indices

    # Run regression without constant
    halflife_model = sm.OLS(delta_spread, spread_lag).fit()
    lambda_val = halflife_model.params[0]

    halflife = -np.log(2) / lambda_val

    print(f"\nHalf-life of mean reversion: {halflife:.2f} days")

    # 4. Suggested Lookback Period for your strategy
    lookback = int(round(halflife))
    print(f"Suggested lookback period (rounded): {lookback} days")

    return lookback

EWA = pd.read_csv('data/EWA_data.csv')
EWC = pd.read_csv('data/EWC_data.csv')
ewa_price = EWA['Close']
ewc_price = EWC['Close']
hedge_ratio = 0.9447
intercept = 0.5892

#suggested lookback period for EWA, EWC is 90 days
# lookback = get_half_life(gold,silver)
lookback = 90

spread = EWC['Close'] - 0.9447*EWA['Close']-0.5892

moving_avg = spread.rolling(lookback).mean()
moving_std = spread.rolling(lookback).std()

z_score = (spread-moving_avg)/(moving_std)

new_df = pd.DataFrame({
    'Date': EWC['Date'],
    'spread': spread,
    'moving_average': moving_avg,
    'moving_std': moving_std,
    'z_score': z_score,
    'ewa_close':EWA['Close'],
    'ewc_close':EWC['Close']
})

new_df.to_csv('results/mean_rev_coint.csv')


entry_threshold = 2.5
exit_threshold = 2

# Initialize signal column
new_df['signal'] = 0

# Entry signals
new_df.loc[new_df['z_score'] > entry_threshold, 'signal'] = -1
new_df.loc[new_df['z_score'] < -entry_threshold, 'signal'] = 1

# Exit signals: override previous signal with 0 when spread reverts
new_df.loc[
    (new_df['z_score'].between(-exit_threshold, exit_threshold)), 
    'signal'
] = 0

# Forward fill to hold positions until exit (simulate trade holding)
new_df['signal'] = new_df['signal'].replace(to_replace=0, method='ffill')

new_df['signal'] = new_df['signal'].shift(1)

# Position sizing
new_df['ewa_position'] = new_df['signal'] * (-hedge_ratio)
new_df['ewc_position'] = new_df['signal'] * 1

# Daily returns of assets
new_df['ewa_return'] = new_df['ewa_close'].pct_change()
new_df['ewc_return'] = new_df['ewc_close'].pct_change()

# Strategy returns from positions
new_df['strategy_returns'] = new_df['ewa_position'].shift(1) * new_df['ewa_return'] + \
                             new_df['ewc_position'].shift(1) * new_df['ewc_return']


new_df = new_df.dropna().copy()
df = new_df

#RETURNS CALCULATION
new_df['strategy_compounded_growth'] = (1 + new_df['strategy_returns']).cumprod()


#worst pain this strategy can give
max_drawdown = ((new_df['strategy_compounded_growth'].cummax() - new_df['strategy_compounded_growth']) / new_df['strategy_compounded_growth'].cummax()).max()


#finding sharpe ratio
mean = df['strategy_returns'].mean()
standard_dev = df['strategy_returns'].std()
sharpe_ratio = (mean / standard_dev) * np.sqrt(252)


#win to loss ratio
wins = (df['strategy_returns']>0).sum()
loss = (df['strategy_returns']<0).sum()
total_trades = (df['signal']!=0).sum()

# total_trades = df['trade_change'].sum()


# Calculate total number of years
start_date = pd.to_datetime(df['Date'].iloc[0])
end_date = pd.to_datetime(df['Date'].iloc[-1])
num_years = (end_date - start_date).days / 365.25


# Strategy CAGR
strategy_cagr = (df['strategy_compounded_growth'].iloc[-1] / df['strategy_compounded_growth'].iloc[0]) ** (1/num_years) - 1



# Print final performance
print('Strategy:', df['strategy_compounded_growth'].iloc[-1])
print('Strategy CAGR:', strategy_cagr)
print('Max Drawdown:',max_drawdown)
print("Sharpe Ratio:",sharpe_ratio)
print('Total Trades:',total_trades)
print('Win Rate:',wins/total_trades)

plt.plot(pd.to_datetime(df['Date']),df['z_score'],label='Strategy Growth', color='blue')
plt.show()










