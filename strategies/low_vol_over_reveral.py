import pandas as pd
from pead_small_caps import calculate_earnings,calculate_portfolio_returns,calculate_portfolio_returns_locked
import matplotlib.pyplot as plt
import numpy as np




def calculate_portfolio_returns_with_slippage(
    df,
    position,
    initial_capital=1000,
    position_percent=10,
    risk_free_rate=0.0,
    commission=0.0,
    max_position_percent=20.0,
    max_portfolio_allocation=85.0,
    # NEW SLIPPAGE PARAMETERS
    slippage_type='spread_based',  # 'fixed_pct', 'fixed_dollar', 'volume_based', 'spread_based'
    slippage_entry=0.05,        # 5 basis points (0.05%) for entry
    slippage_exit=0.05,         # 5 basis points (0.05%) for exit
    bid_ask_spread=0.02,        # 2 basis points typical spread
    market_impact_factor=0.001  # market impact coefficient
):
    """
    Enhanced backtest with multiple slippage models.
    
    Slippage Types:
    - 'fixed_pct': Fixed percentage slippage (most common)
    - 'fixed_dollar': Fixed dollar amount per share
    - 'volume_based': Slippage based on position size vs average volume
    - 'spread_based': Slippage based on bid-ask spread
    """
    
    def calculate_slippage(price, shares, slippage_rate, slippage_type, is_entry=True, volume=None):
        """Calculate slippage based on selected model"""
        
        if slippage_type == 'fixed_pct':
            # Simple percentage slippage
            slippage = price * (slippage_rate / 100)
            
        elif slippage_type == 'fixed_dollar':
            # Fixed dollar amount per share
            slippage = slippage_rate
            
        elif slippage_type == 'volume_based':
            # Slippage increases with position size relative to volume
            if volume is None or volume == 0:
                volume = 1000000  # default volume if not provided
            
            position_value = shares * price
            volume_pct = (shares / volume) * 100 if volume > 0 else 0
            
            # Base slippage + market impact
            base_slippage = price * (slippage_rate / 100)
            market_impact = price * (market_impact_factor * volume_pct)
            slippage = base_slippage + market_impact
            
        elif slippage_type == 'spread_based':
            # Slippage based on bid-ask spread
            spread = price * (bid_ask_spread / 100)
            # Entry: pay half spread + slippage, Exit: pay half spread + slippage
            slippage = (spread / 2) + (price * slippage_rate / 100)
            
        else:
            slippage = 0
            
        return slippage
    
    # --- PREP & INITIALIZATION ---------------------------------------------
    df = df.sort_values(['entry_date', 'ticker']).copy()
    df['shares'] = 0.0
    df['position_value'] = 0.0
    df['dollar_return'] = 0.0
    df['portfolio_value'] = float(initial_capital)
    df['actual_position_percent'] = 0.0
    df['trades_on_entry_date'] = df.groupby('entry_date')['entry_date'].transform('count')
    
    # NEW: Add slippage tracking columns
    df['entry_slippage'] = 0.0
    df['exit_slippage'] = 0.0
    df['total_slippage'] = 0.0
    df['adjusted_entry_price'] = 0.0
    df['adjusted_exit_price'] = 0.0

    current_portfolio = float(initial_capital)
    open_trades = []
    pos_values = []
    total_slippage_cost = 0.0

    # --- LOOP THROUGH TRADES -----------------------------------------------
    for idx, row in df.iterrows():
        entry_date = pd.to_datetime(row['entry_date'])
        exit_date  = pd.to_datetime(row['exit_date'])
        e_price    = row['entry_price']
        x_price    = row['exit_price']
        
        # Get volume if available (for volume-based slippage)
        volume = row.get('volume', None)

        # 1) Free up capital from exited trades
        open_trades = [
            t for t in open_trades
            if pd.to_datetime(t['exit_date']) >= entry_date
        ]

        locked_capital = sum(t['position_value'] for t in open_trades)
        available_capital = current_portfolio * (max_portfolio_allocation/100) - locked_capital

        # 2) Compute sizing
        ideal_size = current_portfolio * (position_percent / 100)
        max_single = current_portfolio * (max_position_percent / 100)
        allocate   = min(ideal_size, max_single, available_capital)

        if allocate <= 0:
            df.at[idx, 'portfolio_value'] = current_portfolio
            continue

        shares = allocate / e_price
        
        # 3) CALCULATE SLIPPAGE
        entry_slippage = calculate_slippage(
            e_price, shares, slippage_entry, slippage_type, True, volume
        )
        exit_slippage = calculate_slippage(
            x_price, shares, slippage_exit, slippage_type, False, volume
        )
        
        # Apply slippage to prices
        if position.lower() == 'l':  # Long position
            adjusted_entry_price = e_price + entry_slippage  # Pay more to buy
            adjusted_exit_price = x_price - exit_slippage    # Receive less to sell
        else:  # Short position
            adjusted_entry_price = e_price - entry_slippage  # Receive less to short
            adjusted_exit_price = x_price + exit_slippage    # Pay more to cover
        
        # Recalculate shares based on adjusted entry price
        shares = allocate / adjusted_entry_price
        actual_position_value = shares * adjusted_entry_price
        
        # Store values
        df.at[idx, 'shares'] = shares
        df.at[idx, 'position_value'] = actual_position_value
        df.at[idx, 'actual_position_percent'] = actual_position_value / current_portfolio * 100
        df.at[idx, 'entry_slippage'] = entry_slippage
        df.at[idx, 'exit_slippage'] = exit_slippage
        df.at[idx, 'adjusted_entry_price'] = adjusted_entry_price
        df.at[idx, 'adjusted_exit_price'] = adjusted_exit_price
        
        pos_values.append(actual_position_value)

        # 4) Calculate P&L with slippage
        if position.lower() == 'l':
            pnl = shares * (adjusted_exit_price - adjusted_entry_price) - 2*commission
        else:
            pnl = shares * (adjusted_entry_price - adjusted_exit_price) - 2*commission

        # Calculate total slippage cost for this trade
        trade_slippage_cost = shares * (entry_slippage + exit_slippage)
        total_slippage_cost += trade_slippage_cost
        
        df.at[idx, 'dollar_return'] = pnl
        df.at[idx, 'total_slippage'] = trade_slippage_cost

        # 5) Record trade and update portfolio
        open_trades.append({
            'exit_date': exit_date,
            'position_value': actual_position_value,
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

    # Sharpe ratio
    excess = df['trade_return_pct'] - risk_free_rate
    sharpe = excess.mean() / excess.std(ddof=1) if excess.std(ddof=1) else np.nan

    # Max drawdown
    pv = df['portfolio_value'].values
    peak = np.maximum.accumulate(pv)
    drawdown = (pv - peak) / peak
    max_drawdown = drawdown.min()

    # Simultaneous trades stats
    max_same_day = df['trades_on_entry_date'].max()
    avg_same_day = df['trades_on_entry_date'].mean()
    days_multi   = (df['trades_on_entry_date'] > 1).sum()

    # --- PRINT RESULTS WITH SLIPPAGE INFO ---------------------------------
    print("="*60)
    print("PORTFOLIO SIMULATION RESULTS (WITH SLIPPAGE)".center(60))
    print("="*60)
    print(f"Initial Capital      : ${initial_capital:,.2f}")
    print(f"Final Capital        : ${current_portfolio:,.2f}")
    print(f"Position Size        : {position_percent:.1f}% of portfolio")
    print(f"Max Single Position  : {max_position_percent:.1f}% of portfolio")
    print(f"Commission/Trade     : ${commission:.2f}")
    print()
    print("SLIPPAGE SETTINGS:")
    print(f"  Slippage Type        : {slippage_type}")
    print(f"  Entry Slippage       : {slippage_entry:.3f}%")
    print(f"  Exit Slippage        : {slippage_exit:.3f}%")
    if slippage_type == 'spread_based':
        print(f"  Bid-Ask Spread       : {bid_ask_spread:.3f}%")
    if slippage_type == 'volume_based':
        print(f"  Market Impact Factor : {market_impact_factor:.4f}")
    print(f"  Total Slippage Cost  : ${total_slippage_cost:,.2f}")
    print(f"  Slippage as % of P&L : {(total_slippage_cost / abs(current_portfolio - initial_capital)) * 100:.2f}%")
    print()
    print("POSITION SIZE EVOLUTION:")
    if pos_values:
        print(f"  First Trade Size     : ${pos_values[0]:,.2f}")
        print(f"  Last Trade Size      : ${pos_values[-1]:,.2f}")
        print(f"  Average Trade Size   : ${np.mean(pos_values):,.2f}")
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

    # --- ENHANCED PLOTTING WITH SLIPPAGE ----------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), tight_layout=True)
    
    # Portfolio value over time
    axes[0,0].plot(pd.to_datetime(df['exit_date']), df['portfolio_value'], marker='o', linewidth=2)
    axes[0,0].axhline(initial_capital, linestyle='--', alpha=0.7, label='Initial Capital')
    axes[0,0].set_title('Portfolio Value Over Time')
    axes[0,0].set_xlabel('Exit Date')
    axes[0,0].set_ylabel('Portfolio Value ($)')
    axes[0,0].legend()
    axes[0,0].grid(alpha=0.3)
    
    # Individual trade P&L
    colors = ['green' if x>0 else 'red' for x in df['dollar_return']]
    axes[0,1].bar(range(total_trades), df['dollar_return'], color=colors, alpha=0.7)
    axes[0,1].axhline(0, linestyle='-', alpha=0.8)
    axes[0,1].set_title('Individual Trade P&L')
    axes[0,1].set_xlabel('Trade #')
    axes[0,1].set_ylabel('P&L ($)')
    axes[0,1].grid(alpha=0.3)
    
    # Slippage cost per trade
    axes[1,0].bar(range(total_trades), df['total_slippage'], color='orange', alpha=0.7)
    axes[1,0].set_title('Slippage Cost Per Trade')
    axes[1,0].set_xlabel('Trade #')
    axes[1,0].set_ylabel('Slippage Cost ($)')
    axes[1,0].grid(alpha=0.3)
    
    # Cumulative slippage cost
    axes[1,1].plot(range(total_trades), df['total_slippage'].cumsum(), color='red', linewidth=2)
    axes[1,1].set_title('Cumulative Slippage Cost')
    axes[1,1].set_xlabel('Trade #')
    axes[1,1].set_ylabel('Cumulative Slippage ($)')
    axes[1,1].grid(alpha=0.3)

    plt.show()

    return df


# Example usage with different slippage models:
"""
# Fixed percentage slippage (most common)
df_result = calculate_portfolio_returns_with_slippage(
    df, 'l', slippage_type='fixed_pct', slippage_entry=0.05, slippage_exit=0.05
)

# Volume-based slippage (more realistic for large positions)
df_result = calculate_portfolio_returns_with_slippage(
    df, 'l', slippage_type='volume_based', slippage_entry=0.03, slippage_exit=0.03,
    market_impact_factor=0.001
)

# Spread-based slippage (most realistic)
df_result = calculate_portfolio_returns_with_slippage(
    df, 'l', slippage_type='spread_based', slippage_entry=0.02, slippage_exit=0.02,
    bid_ask_spread=0.02
)
"""







def overnight_reversal_for_micro(ticker_df,folder='micro_caps_stock_data',ra_con=10,reversal_end=1):
    results_list = []
    ticker_list = ticker_df['Symbol'].to_list()

    big_df = pd.read_parquet(f'{folder}/big_boy.parquet')
    grouped = big_df.groupby('Ticker')

    for ticker, stock_price_series in grouped:
        if ticker not in ticker_list:
            continue

        stock_price_series = stock_price_series.sort_values('Date')

        stock_price_series = stock_price_series.reset_index(drop=True)

        stock_price_series['moving_vol_for_past'] = stock_price_series['Volume'].rolling(ra_con).mean().shift(1)
        stock_price_series['vol_change'] = (stock_price_series['Volume'] - stock_price_series['moving_vol_for_past']) / stock_price_series['moving_vol_for_past']

        stock_price_series['yday_close'] = stock_price_series['Close'].shift(2)
        stock_price_series['daily_return'] = (stock_price_series['Close'].shift(1) - stock_price_series['yday_close']) / stock_price_series['yday_close']

        stock_price_series['signal'] = 0
        stock_price_series.loc[
            (stock_price_series['vol_change'] < 0.5) &
            (stock_price_series['daily_return'] < -0.05)&(stock_price_series['daily_return']>-0.1)&
            (stock_price_series['vol_change']>0.2),
            #,
            'signal'
        ] = 1

        # stock_price_series['signal'] = stock_price_series['signal'].shift(1)

        positions = stock_price_series[stock_price_series['signal'] == 1]

        for idx, row in positions.iterrows():
            if idx + reversal_end >= len(stock_price_series):
                print(f'Not enough rows for {ticker} at date {row["Date"]}')
                continue

            exit_row_ohcl = stock_price_series.iloc[idx + reversal_end]

            results_list.append({
                'entry_date': row['Date'],
                'exit_date': exit_row_ohcl['Date'],
                'entry_price': row['Close'],
                'exit_price': exit_row_ohcl['Open'],
                'ticker': ticker,
                'vol_change':row['vol_change'],
                'daily_return': row['daily_return']


            })

    results = pd.DataFrame(results_list)
    results.to_parquet('results/low_vol_over_reversal.parquet', index=False)




# Example usage with conservative sniper settings
"""
# Ultra-conservative sniper approach
results = overnight_reversal_sniper(
    ticker_df,
    min_daily_return=-0.075,        # Tight range
    max_daily_return=-0.065,        
    min_vol_surge=1.2,              # Significant volume
    max_vol_surge=2.5,              
    rsi_oversold_threshold=20,      # Very oversold
    max_trades_per_day=2,           # Very selective
    quality_score_threshold=0.8     # High quality only
)

# Moderate sniper approach  
results = overnight_reversal_sniper(
    ticker_df,
    min_daily_return=-0.08,         
    max_daily_return=-0.06,         
    min_vol_surge=0.8,              
    max_vol_surge=3.0,              
    rsi_oversold_threshold=25,      
    max_trades_per_day=3,           
    quality_score_threshold=0.7     
)
"""






if __name__=='__main__':
    df = pd.read_csv('micro_caps.csv')
    overnight_reversal_for_micro(df,folder='micro_caps_stock_data')



    results = pd.read_parquet('results/low_vol_over_reversal.parquet')
    calculate_portfolio_returns_with_slippage(results,position='l')



