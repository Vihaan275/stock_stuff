import yfinance as yf
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from strategies.futures_mean_rev import simple_mean_rev
from strategies.futures_simple_rolling_avg import simple_rolling_avg
from strategies.reddit_simple import test_run
from strategies.republican_strat import repub_years


#ES! Daily bars are from yesterday 6pm to today 5pm
#SPY daily bars are from today 9:30am to today 4pm


# Load data
old_df = pd.read_csv('test.csv')
spy_data = pd.read_csv('data/spy_data.csv')

# Apply strategy function
df = repub_years(spy_data)


type_shi = 'close_to_close_returns'

if type_shi == 'close_to_close_returns':
    df['signal'] = df['signal'].shift(1)

# Compute returns outside the function
df['open_to_close_returns'] = (df['Close']-df['Open'])/df['Open']  # Daily returns
df['close_to_close_returns'] = df['Close'].pct_change()
df['strategy_returns'] = df['signal']*df[type_shi]  # Avoid lookahead bias, shift should always be 1

df['spy_close_to_close'] = spy_data['Close'].pct_change()

# Compute compounded growth
df['strategy_compounded_growth'] = (1 + df['strategy_returns']).cumprod()

df['regular_compounded_growth'] = (1 + df['close_to_close_returns']).cumprod()

# Save results
df.to_csv('results.csv', index=False)


#finding correlation for makret neutral strategies
correlation = df['strategy_returns'].corr(df[type_shi])

#worst pain this strategy can give
max_drawdown = ((df['strategy_compounded_growth'].cummax() - df['strategy_compounded_growth']) / df['strategy_compounded_growth'].cummax()).max()


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

# Buy and Hold CAGR
buy_and_hold_cagr = (df['regular_compounded_growth'].iloc[-1] / df['regular_compounded_growth'].iloc[0]) ** (1/num_years) - 1

# Strategy CAGR
strategy_cagr = (df['strategy_compounded_growth'].iloc[-1] / df['strategy_compounded_growth'].iloc[0]) ** (1/num_years) - 1



# Print final performance
print('Buy and Hold:', df['regular_compounded_growth'].iloc[-1])
print('Strategy:', df['strategy_compounded_growth'].iloc[-1])
print('Buy and Hold CAGR:', buy_and_hold_cagr)
print('Strategy CAGR:', strategy_cagr)
print('Correlation:',correlation)
print('Max Drawdown:',max_drawdown)
print("Sharpe Ratio:",sharpe_ratio)
print('Total Trades:',total_trades)
print('Win Rate:',wins/total_trades)

plt.plot(pd.to_datetime(df['Date']),df['strategy_compounded_growth'], marker='o',label='Strategy Growth', color='blue')
plt.show()
