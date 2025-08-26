import csv
import pandas as pd
import numpy as np
from alpha_research.pairs import pairs
import matplotlib.pyplot as plt


import matplotlib.pyplot as plt

def graph_strat_returns(df):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot strategy compounded growth on the primary y-axis
    ax1.plot(df['strategy_compounded_growth'], label='Strategy Growth', color='blue')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Compounded Growth', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')



    # Title and layout
    plt.title('Strategy Compounded Returns and Volatility Over Time')
    fig.tight_layout()

    # Combined legend (optional but recommended)
    lines_1, labels_1 = ax1.get_legend_handles_labels()

    plt.show()

   

gold = pd.read_csv('./data/gold_yahoo.csv')
silver = pd.read_csv('./data/silver_yahoo.csv')

df = pairs(gold,silver,num=3,std_away=0.3,bounds=0.2,window=15)
#3,1.1 gives 1.75 returns
#3,1.08 gives 2.53 returns
#1.06 gives 2.66, with 0.6 
#1.06 with +- 0.9 stds for exit gives 3.175x with normal z-score
#coint: 0.4,0.37,15 gives 3.69x with 0.561 sharpe
#coint: 0.3,0.2,15 gives 5x with 0.73 sharpe
#coint: 0.3,0.25,15 gives 5.33x with 0.74 sharpe



signal = 'exit_identified_signal'

#coint_raw_signal
#exit_identified_signal


df['gold_returns'] = gold['close'].pct_change()
df['silver_returns'] = silver['close'].pct_change()

df['strategy_returns'] = 0
df.loc[(df[signal]==1) ,'strategy_returns'] = df['gold_returns']-df['silver_returns']
df.loc[(df[signal]==-1),'strategy_returns'] = -df['gold_returns']+df['silver_returns']

df['strategy_compounded_growth'] = (1 + df['strategy_returns']).cumprod()

#worst pain this strategy can give
max_drawdown = ((df['strategy_compounded_growth'].cummax() - df['strategy_compounded_growth']) / df['strategy_compounded_growth'].cummax()).max()

df.to_csv('data/Gold_Silver_results.csv')

#finding sharpe ratio
mean = df['strategy_returns'].mean()
standard_dev = df['strategy_returns'].std()
sharpe_ratio = (mean / standard_dev) * np.sqrt(252)


#win to loss ratio
wins = (df['strategy_returns']>0).sum()
loss = (df['strategy_returns']<0).sum()
total_trades = (df[signal]!=0).sum()



# Print final performance
print('Strategy:', df['strategy_compounded_growth'].iloc[-1])
print('Max Drawdown:',max_drawdown)
print("Sharpe Ratio:",sharpe_ratio)
print('total: ',total_trades)
print('wins: ',wins)
print('Losses: ',loss)
print('Winning % is ',(wins/total_trades)*100)

graph_strat_returns(df)