import pandas as pd
import yfinance as yf
import datetime
import os
import matplotlib.pyplot as plt
import numpy as np

#Basically find earnings dates
#invest a day after
#

# ticker_list = ticker_df['Symbol'].to_list()


def get_price_series(ticker,folder):

    main_df = pd.read_csv('7/stocks_latest/stock_prices_latest.csv')

    try:
        stock_price_series = pd.read_csv(f'{folder}/{ticker}.csv')#taking the price series
    except Exception as e:
        print(f"Price data for {ticker} not found in {folder}, skipping.")
        print(e)

#gives filtered earnings_lates.csv
def get_earning_dates(ticker,earning_df):
    new_df = earning_df.loc[earning_df['symbol']== ticker]

    if new_df is None or new_df.empty:
        return 'no'

    return new_df


def make_stop_loss():
    pass

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_portfolio_returns_locked(
    df,
    position,
    initial_capital=1000,
    position_percent=10,
    risk_free_rate=0.0,
    commission=0.0,
    max_position_percent=20.0,
    max_portfolio_allocation=85.0
):
    """
    Backtest with locked-capital constraint: no new trade can use funds already
    deployed in any open position.
    Prints full suite of strategy metrics and plots P&L.
    """
    # --- PREP & INITIALIZATION ---------------------------------------------
    df = df.sort_values(['entry_date', 'ticker']).copy()
    # ensure float columns
    df['shares'] = 0.0
    df['position_value'] = 0.0
    df['dollar_return'] = 0.0
    df['portfolio_value'] = float(initial_capital)
    df['actual_position_percent'] = 0.0
    df['trades_on_entry_date'] = df.groupby('entry_date')['entry_date'].transform('count')

    current_portfolio = float(initial_capital)
    open_trades = []   # each: {'exit_date': date, 'position_value': float, 'pnl': float}

    # for position‑size evolution
    pos_values = []

    # --- LOOP THROUGH TRADES -----------------------------------------------
    for idx, row in df.iterrows():
        entry_date = pd.to_datetime(row['entry_date'])
        exit_date  = pd.to_datetime(row['exit_date'])
        e_price    = row['entry_price']
        x_price    = row['exit_price']

        # 1) free up capital from any trades that have exited before today
        open_trades = [
            t for t in open_trades
            if pd.to_datetime(t['exit_date']) >= entry_date
        ]

        locked_capital = sum(t['position_value'] for t in open_trades)
        available_capital = current_portfolio * (max_portfolio_allocation/100) - locked_capital

        # 2) compute sizing
        ideal_size = current_portfolio * (position_percent / 100)
        max_single = current_portfolio * (max_position_percent / 100)
        allocate   = min(ideal_size, max_single, available_capital)

        if allocate <= 0:
            # skip if no capital
            df.at[idx, 'portfolio_value'] = current_portfolio
            continue

        shares = allocate / e_price
        df.at[idx, 'shares'] = shares
        df.at[idx, 'position_value'] = allocate
        df.at[idx, 'actual_position_percent'] = allocate / current_portfolio * 100
        pos_values.append(allocate)

        # 3) P&L
        if position.lower() == 'l':
            pnl = shares * (x_price - e_price) - 2*commission
        else:
            pnl = shares * (e_price - x_price) - 2*commission

        df.at[idx, 'dollar_return'] = pnl

        # 4) record open trade & immediately update portfolio
        open_trades.append({
            'exit_date': exit_date,
            'position_value': allocate,
            'pnl': pnl
        })
        current_portfolio += pnl
        df.at[idx, 'portfolio_value'] = current_portfolio

    # --- POST-PROCESS METRICS ----------------------------------------------
    df['trade_return_pct'] = df['dollar_return'] / df['position_value'].replace(0, np.nan)

    total_trades = len(df)
    winning_trades = (df['dollar_return'] > 0).sum()
    win_rate = winning_trades / total_trades if total_trades else np.nan

    total_return = (current_portfolio - initial_capital) / initial_capital
    avg_return_per_trade = df['dollar_return'].mean()
    avg_return_pct = df['trade_return_pct'].mean()

    gross_profit = df.loc[df['dollar_return'] > 0, 'dollar_return'].sum()
    gross_loss   = -df.loc[df['dollar_return'] < 0, 'dollar_return'].sum()
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    # Sharpe
    excess = df['trade_return_pct'] - risk_free_rate
    sharpe = excess.mean() / excess.std(ddof=1) if excess.std(ddof=1) else np.nan

    # Max drawdown
    pv = df['portfolio_value'].values
    peak = np.maximum.accumulate(pv)
    drawdown = (pv - peak) / peak
    max_drawdown = drawdown.min()

    # Simultaneous trades stats (by entry_date grouping)
    max_same_day = df['trades_on_entry_date'].max()
    avg_same_day = df['trades_on_entry_date'].mean()
    days_multi   = (df['trades_on_entry_date'] > 1).sum()

    # --- PRINT RESULTS -----------------------------------------------------
    print("="*60)
    print("PORTFOLIO SIMULATION RESULTS".center(60))
    print("="*60)
    print(f"Initial Capital      : ${initial_capital:,.2f}")
    print(f"Final Capital        : ${current_portfolio:,.2f}")
    print(f"Position Size        : {position_percent:.1f}% of portfolio")
    print(f"Max Single Position  : {max_position_percent:.1f}% of portfolio")
    print(f"Commission/Trade     : ${commission:.2f}")
    print()
    print("POSITION SIZE EVOLUTION:")
    if pos_values:
        print(f"  First Trade Size     : ${pos_values[0]:,.2f}")
        print(f"  Last Trade Size      : ${pos_values[-1]:,.2f}")
        print(f"  Average Trade Size   : ${np.mean(pos_values):,.2f}")
    else:
        print("  (no trades entered)")
    print()
    print("SIMULTANEOUS TRADE HANDLING:")
    print(f"  Max Trades Same Day  : {int(max_same_day)}")
    print(f"  Avg Trades Per Day   : {avg_same_day:.1f}")
    print(f"  Days w/ Multiple     : {days_multi}")
    print()
    print("TRADE STATISTICS:")
    print(f"  Total Trades         : {total_trades}")
    print(f"  Winning Trades       : {winning_trades}")
    print(f"  Win Rate             : {win_rate*100:.2f}%")
    print()
    print("RETURN METRICS:")
    print(f"  Total Return         : {total_return*100:.2f}%")
    print(f"  Avg Return/Trade     : ${avg_return_per_trade:.2f}")
    print(f"  Avg Return %/Trade   : {avg_return_pct*100:.2f}%")
    print(f"  Gross Profit         : ${gross_profit:,.2f}")
    print(f"  Gross Loss           : ${gross_loss:,.2f}")
    print(f"  Profit Factor        : {profit_factor:.2f}")
    print()
    print("RISK METRICS:")
    print(f"  Sharpe Ratio         : {sharpe:.2f}")
    print(f"  Max Drawdown         : {max_drawdown*100:.2f}%")
    print("="*60)

    # --- PLOTTING ----------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), tight_layout=True)
    # Portfolio value over time
    ax1.plot(pd.to_datetime(df['exit_date']), df['portfolio_value'], marker='o')
    ax1.axhline(initial_capital, linestyle='--', alpha=0.7, label='Initial Capital')
    ax1.set_title('Portfolio Value Over Time')
    ax1.set_xlabel('Exit Date')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend()
    ax1.grid(alpha=0.3)
    # Individual trade P&L
    colors = ['green' if x>0 else 'red' for x in df['dollar_return']]
    ax2.bar(range(total_trades), df['dollar_return'], color=colors, alpha=0.7)
    ax2.axhline(0, linestyle='-', alpha=0.8)
    ax2.set_title('Individual Trade P&L')
    ax2.set_xlabel('Trade #')
    ax2.set_ylabel('P&L ($)')
    ax2.grid(alpha=0.3)

    plt.show()

    return df




def calculate_portfolio_returns(df, position, initial_capital=1000, position_percent=10, 
                              risk_free_rate=0.0, commission=0.0, max_position_percent=20.0,
                              max_portfolio_allocation=100.0):
    """
    Calculate performance metrics using percentage-based position sizing that handles simultaneous trades.
    
    Args:
        df: DataFrame with entry_price, exit_price, entry_date, exit_date
        position: 'l' for long, 's' for short
        initial_capital: Starting portfolio value
        position_percent: Percentage of current portfolio value to allocate per trade
        risk_free_rate: Risk-free rate for Sharpe calculation
        commission: Commission per trade (dollar amount)
        max_position_percent: Maximum percentage of portfolio for any single trade
        max_portfolio_allocation: Maximum total percentage of portfolio that can be allocated
    """
    
    # Sort by entry date, then by ticker for consistent ordering
    df = df.sort_values(['entry_date', 'ticker']).copy()
    
    # Initialize portfolio tracking
    df['shares'] = 0.0
    df['dollar_return'] = 0.0
    df['portfolio_value'] = initial_capital
    df['position_value'] = 0.0
    df['actual_position_percent'] = 0.0
    df['trades_same_day'] = 0
    
    current_portfolio_value = initial_capital
    
    # Group trades by entry date to handle simultaneous trades
    for entry_date, day_group in df.groupby('entry_date'):
        trades_today = len(day_group)
        
        # Calculate available capital for this day
        available_capital = current_portfolio_value * (max_portfolio_allocation / 100)
        
        # Calculate ideal position size per trade
        ideal_position_size = current_portfolio_value * (position_percent / 100)
        max_single_position = current_portfolio_value * (max_position_percent / 100)
        
        # If we have multiple trades, we need to allocate properly
        if trades_today == 1:
            # Single trade - use normal allocation
            position_size = min(ideal_position_size, max_single_position, available_capital)
        else:
            # Multiple trades - distribute available capital
            total_ideal_allocation = trades_today * ideal_position_size
            
            if total_ideal_allocation <= available_capital:
                # We can afford all trades at ideal size
                position_size = min(ideal_position_size, max_single_position)
            else:
                # Scale down positions to fit available capital
                scaling_factor = available_capital / total_ideal_allocation
                position_size = min(ideal_position_size * scaling_factor, max_single_position)
        
        # Apply position sizing to all trades on this day
        day_total_allocation = 0
        for idx in day_group.index:
            row = df.loc[idx]
            
            # Final position size (ensure we don't exceed available capital)
            remaining_capital = available_capital - day_total_allocation
            final_position_size = min(position_size, remaining_capital)
            
            if final_position_size <= 0:
                # No more capital available for this trade
                df.at[idx, 'shares'] = 0
                df.at[idx, 'position_value'] = 0
                df.at[idx, 'dollar_return'] = 0
                df.at[idx, 'actual_position_percent'] = 0
                continue
            
            # Calculate shares and track allocation
            shares = final_position_size / row['entry_price']
            day_total_allocation += final_position_size
            
            df.at[idx, 'shares'] = shares
            df.at[idx, 'position_value'] = final_position_size
            df.at[idx, 'actual_position_percent'] = (final_position_size / current_portfolio_value) * 100
            df.at[idx, 'trades_same_day'] = trades_today
            
            # Calculate dollar P&L
            if position == 'l':  # Long position
                dollar_pnl = shares * (row['exit_price'] - row['entry_price']) - (2 * commission)
            elif position == 's':  # Short position  
                dollar_pnl = shares * (row['entry_price'] - row['exit_price']) - (2 * commission)
            
            df.at[idx, 'dollar_return'] = dollar_pnl
        
        # Update portfolio value after all trades for this day
        day_pnl = sum(df.at[idx, 'dollar_return'] for idx in day_group.index)
        current_portfolio_value += day_pnl
        
        # Set portfolio value for all trades on this day
        for idx in day_group.index:
            df.at[idx, 'portfolio_value'] = current_portfolio_value
    
    # Calculate portfolio-level metrics
    df['portfolio_return'] = (df['portfolio_value'] - initial_capital) / initial_capital
    df['trade_return_pct'] = df['dollar_return'] / df['position_value']  # Return per trade as %
    
    # Performance metrics
    total_trades = len(df)
    winning_trades = (df['dollar_return'] > 0).sum()
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    # Total return
    total_return = (current_portfolio_value - initial_capital) / initial_capital
    
    # Average return per trade
    avg_return_per_trade = df['dollar_return'].mean()
    avg_return_pct = df['trade_return_pct'].mean()
    
    # Sharpe ratio based on trade returns
    excess_returns = df['trade_return_pct'] - risk_free_rate
    sharpe_ratio = excess_returns.mean() / excess_returns.std(ddof=1) if excess_returns.std(ddof=1) != 0 else np.nan
    
    # Max drawdown from peak portfolio value
    portfolio_values = df['portfolio_value'].values
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - peak) / peak
    max_drawdown = np.min(drawdown)
    
    # Profit factor
    gross_profit = df[df['dollar_return'] > 0]['dollar_return'].sum()
    gross_loss = abs(df[df['dollar_return'] < 0]['dollar_return'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    
    # Print comprehensive stats
    print("=" * 50)
    print("PORTFOLIO SIMULATION RESULTS")
    print("=" * 50)
    print(f"Initial Capital      : ${initial_capital:,.2f}")
    print(f"Final Capital        : ${current_portfolio_value:,.2f}")
    print(f"Position Size        : {position_percent:.1f}% of portfolio")
    print(f"Max Position Size    : {max_position_percent:.1f}% of portfolio")
    print(f"Commission/Trade     : ${commission:.2f}")
    print()
    print("POSITION SIZE EVOLUTION:")
    print(f"First Trade Size     : ${df['position_value'].iloc[0]:,.2f} ({df['actual_position_percent'].iloc[0]:.1f}%)")
    print(f"Last Trade Size      : ${df['position_value'].iloc[-1]:,.2f} ({df['actual_position_percent'].iloc[-1]:.1f}%)")
    print(f"Average Trade Size   : ${df['position_value'].mean():,.2f}")
    print()
    print("SIMULTANEOUS TRADE HANDLING:")
    max_same_day = df['trades_same_day'].max()
    avg_same_day = df['trades_same_day'].mean()
    days_with_multiple = (df['trades_same_day'] > 1).sum()
    print(f"Max Trades Same Day  : {max_same_day}")
    print(f"Avg Trades Per Day   : {avg_same_day:.1f}")
    print(f"Days w/ Multiple     : {days_with_multiple}")
    
    # Show some examples of position scaling
    multi_trade_days = df[df['trades_same_day'] > 1]
    if not multi_trade_days.empty:
        print(f"Example Multi-Trade Day:")
        example_date = multi_trade_days['entry_date'].iloc[0]
        example_trades = df[df['entry_date'] == example_date]
        print(f"  Date: {example_date}")
        print(f"  Trades: {len(example_trades)}")
        print(f"  Total Allocation: {example_trades['actual_position_percent'].sum():.1f}%")
    print()
    print("TRADE STATISTICS:")
    print(f"Total Trades       : {total_trades}")
    print(f"Winning Trades     : {winning_trades}")
    print(f"Win Rate           : {win_rate*100:.2f}%")
    print()
    print("RETURN METRICS:")
    print(f"Total Return       : {total_return*100:.2f}%")
    print(f"Avg Return/Trade   : ${avg_return_per_trade:.2f}")
    print(f"Avg Return %/Trade : {avg_return_pct*100:.2f}%")
    print(f"Gross Profit       : ${gross_profit:.2f}")
    print(f"Gross Loss         : ${gross_loss:.2f}")
    print(f"Profit Factor      : {profit_factor:.2f}")
    print()
    print("RISK METRICS:")
    print(f"Sharpe Ratio       : {sharpe_ratio:.2f}")
    print(f"Max Drawdown       : {max_drawdown*100:.2f}%")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Portfolio value over time
    ax1.plot(pd.to_datetime(df['exit_date']), df['portfolio_value'], 
             marker='o', linewidth=2, markersize=4)
    ax1.axhline(y=initial_capital, color='r', linestyle='--', alpha=0.7, label='Initial Capital')
    ax1.set_title('Portfolio Value Over Time')
    ax1.set_xlabel('Exit Date')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)
    
    # Individual trade returns
    colors = ['green' if x > 0 else 'red' for x in df['dollar_return']]
    ax2.bar(range(len(df)), df['dollar_return'], color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.8)
    ax2.set_title('Individual Trade P&L')
    ax2.set_xlabel('Trade Number')
    ax2.set_ylabel('P&L ($)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return df

def post_earnings_drift(ticker_df, gap=1,days=3, surprise_percent=5,max=7, folder='micro_caps_stock_data',atr_num = 5,vol_num = 20):
    pead_results = pd.DataFrame(columns=['entry_date', 'exit_date', 'entry_price', 'exit_price','ticker','Surprise(%)'])

    main_earning_dates = pd.read_csv('modified_earning_dates.csv',low_memory=False)
    main_earning_dates['date'] = pd.to_datetime(main_earning_dates['date'])

    ticker_list = ticker_df['Symbol'].to_list()

    for i in ticker_list:
        try:
            stock_price_series = pd.read_csv(f'{folder}/{i}.csv')#taking the price series
        except Exception as e:
            print(f"Price data for {i} not found, skipping.")
            print(e)
            continue
        stock_price_series['Date'] = pd.to_datetime(stock_price_series['Date'])
        stock_price_series['Date'] = stock_price_series['Date'].dt.date#idk why, but we need this line
        
        stock_price_series['ATR'] = (stock_price_series['High'].rolling(atr_num).max() - stock_price_series['Low'].rolling(atr_num).min()).shift(1)
        stock_price_series['rel_vol'] = stock_price_series['Volume']/(stock_price_series['Volume'].shift(1).rolling(window=vol_num).mean())
        
        #gets filtered earning dates
        earning_dates = get_earning_dates(i,main_earning_dates)

        if isinstance(earning_dates, str) and earning_dates == 'no': #change this so that it can handle stocks that aren't in the csv
            continue               #currently, it cannot use those stocks, but we need to make a yfinance ticker for that


        if earning_dates is None or earning_dates.empty:
            print(f"No earnings data for {i}, skipping.")
            continue
        
        earning_dates_date_list = earning_dates['date'].to_list()

        for date in earning_dates_date_list:
            try:
                # This handles both single and multiple entries per date
                earning_date_row = earning_dates.loc[earning_dates['date'] == date]
                surprise_val = earning_date_row['surprise(%)'].item()
                if isinstance(surprise_val, pd.Series):
                    surprise = surprise_val.iloc[0]
                else:
                    surprise = surprise_val
            except KeyError:
                print(f"Surprise not found for {i} on {date}")
                continue

            if pd.isna(surprise) or surprise is None:
                continue

            if surprise > surprise_percent and max>surprise:
                if date.time()>datetime.time(hour=4):

                    # Try to find the row that matches the earnings date
                    match_date = date.date()

                    earning_date_ohcl = stock_price_series[stock_price_series['Date'] == match_date]

                    if earning_date_ohcl.empty:
                        print(f"No matching date in price data for {i} on {match_date}")
                        continue

                    earning_date_index = earning_date_ohcl.index[0]
                    earning_date_index_pos = stock_price_series.index.get_loc(earning_date_index)

                    if 0 <= earning_date_index_pos + gap < len(stock_price_series):
                        entry_row = stock_price_series.iloc[earning_date_index_pos + gap]
                    else:
                        # handle the out of bounds situation
                        # e.g., skip this ticker/date or log a warning
                        print(f"Index out of bounds for {i} on earnings date {match_date}. Skipping.")
                        continue

                    

                    if earning_date_index_pos + days >= len(stock_price_series):
                        print(f"Not enough data after earnings date for {i} on {match_date}")
                        continue

                    if 5>earning_date_ohcl['rel_vol'].item()>3 and earning_date_ohcl['ATR'].item()<0.5:
                        entry_row = stock_price_series.iloc[earning_date_index_pos + gap]

                        exit_row = stock_price_series.iloc[earning_date_index_pos + gap+days]

                        if days==0:
                            pead_results = pd.concat([
                            pead_results,
                            pd.DataFrame([{
                                'entry_date': entry_row['Date'],
                                'exit_date': entry_row['Date'],
                                'entry_price': entry_row['Open'],
                                'exit_price': entry_row['Close'],
                                'ticker':i,
                                'Surprise(%)':surprise,
                                'release_time':earning_date_row['release_time'].item(),
                                'ATR': entry_row['ATR'],
                                'rel_vol':entry_row['rel_vol']
                            }])
                        ], ignore_index=True)
                            

                        else:
                            pead_results = pd.concat([
                                pead_results,
                                pd.DataFrame([{
                                    'entry_date': entry_row['Date'],
                                    'exit_date': exit_row['Date'],
                                    'entry_price': entry_row['Open'],
                                    'exit_price': exit_row['Open'],
                                    'ticker':i,
                                    'Surprise(%)':surprise,
                                    'release_time':earning_date_row['release_time'].item(),
                                    'ATR': entry_row['ATR'],
                                    'rel_vol':entry_row['rel_vol']
                                }])
                            ], ignore_index=True)

    pead_results.to_csv('pead_results.csv', index=False)



def calculate_earnings(df, position,risk_free_rate=0.0):
    """,
    Calculate performance metrics for a PEAD short strategy.
    
    Assumes entry/exit columns exist and are from short trades:
    return = (entry_price - exit_price) / entry_price
    """

    # Sort by exit date to preserve temporal order
    df = df.sort_values('exit_date').copy()

    # Per-trade return for short strategy
    if position == 'l':
        df['return'] = (df['exit_price'] - df['entry_price']) / df['entry_price']
    elif position=='s':
        df['return'] = (df['entry_price'] - df['exit_price']) / df['entry_price']

    # Cumulative return (compounding)
    df['cumulative_return'] = (1 + df['return']).cumprod() - 1

    # Win rate (% of trades with positive return)
    total_trades = len(df)
    winning_trades = (df['return'] > 0).sum()
    win_rate = winning_trades / total_trades if total_trades > 0 else 0

    # Sharpe Ratio (assume per-trade returns if daily data not available)
    excess_returns = df['return'] - risk_free_rate
    sharpe_ratio = excess_returns.mean() / excess_returns.std(ddof=1) if excess_returns.std(ddof=1) != 0 else np.nan

    # Max Drawdown
    cumulative = (1 + df['return']).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    # Print stats
    print(f"Total Trades       : {total_trades}")
    print(f"Win Rate           : {win_rate*100:.2f}%")
    print(f"Sharpe Ratio       : {sharpe_ratio:.2f}")
    print(f"Max Drawdown       : {max_drawdown*100:.2f}%")
    print(f"Final Return       : {(df['cumulative_return'].iloc[-1])*100:.2f}%")

    # Plot cumulative return
    plt.figure(figsize=(10,6))
    plt.plot(pd.to_datetime(df['exit_date']), df['cumulative_return'], marker='o', label='Cumulative Return')
    plt.title('Cumulative Return of Short Strategy Over Time')
    plt.xlabel('Exit Date')
    plt.ylabel('Cumulative Return')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    df = pd.read_csv('small_caps.csv')
    # post_earnings_drift(df,days=3,surprise_percent=-25,max=50,folder='small_cap_stock_data')    






    #days = 3, 4&7% had pretty good returns

    results = pd.read_csv('pead_results.csv')
    calculate_portfolio_returns_locked(df=results,position='l',position_percent=50,max_position_percent=50)
