import pandas as pd
from ta.momentum import RSIIndicator
import datetime,time
import numpy as np

def make_trade_log(
    df,
    stop_loss_pct: float = 0.02,
    spread: float = 0.0015/2,
    market_close_hour: int = 12,
    market_close_minute: int = 00,
    position: str = 'long'
):
    """
    Generates a trade log for simultaneous pair trades on BOIL and KOLD.

    Args:
        df (pd.DataFrame): Merged DataFrame with timestamped OHLC for both symbols.
        stop_loss_pct (float): Stop-loss percentage (e.g., 0.02 for 2%).
        spread (float): Round-trip spread in price units.
        market_close_hour (int): Hour of daily close (24h).
        market_close_minute (int): Minute of daily close.
        position (str): 'long' or 'short' to trade both symbols accordingly.

    Returns:
        pd.DataFrame: Trade entries, exits, and returns.
    """
    results = []
    in_trade = False
    entry_date = None
    current_position_date = None

    half_spread = spread /2

    df['Date'] = pd.to_datetime(df['Date'])

    for _, row in df.iterrows():
        ts = row['Date']

        # ENTRY: signal + before market close
        if not in_trade and row['boil_signal'] != 0 and ts.time() < datetime.time(market_close_hour, market_close_minute):
        
            entry_date = ts.date()
            if any(r['entry_date'] == entry_date for r in results):
                continue  # skip if already entered on this date
            
            in_trade = True
            

            entry_time = ts.time()
            # Compute exit timestamp for EOD with same tz as ts
            exit_time = pd.Timestamp(
                f"{entry_date} {market_close_hour:02d}:{market_close_minute:02d}",
                tz=ts.tzinfo
            )

            # Entry prices with half-spread adjustment
            if position == 'long':
                entry_price_boil = row['Open_boil'] + (half_spread*row['Open_boil'])
                entry_price_kold = row['Open_kold'] + (half_spread*row['Open_kold'])
                boil_stop = entry_price_boil * (1 - stop_loss_pct)
                kold_stop = entry_price_kold * (1 - stop_loss_pct)
            else:  # short
                entry_price_boil = row['Open_boil'] - (half_spread*row['Open_boil'])
                entry_price_kold = row['Open_kold'] - (half_spread*row['Open_kold'])
                boil_stop = entry_price_boil * (1 + stop_loss_pct)
                kold_stop = entry_price_kold * (1 + stop_loss_pct)

            continue

        # DURING TRADE: same day before exit_time
        if in_trade and ts.date() == entry_date and ts < exit_time:
            boil_hit = (row['Low_boil'] <= boil_stop) if position == 'long' else (row['High_boil'] >= boil_stop)
            kold_hit = (row['Low_kold'] <= kold_stop) if position == 'long' else (row['High_kold'] >= kold_stop)

            if boil_hit and kold_hit:
                if position == 'long':
                    exit_price_boil = boil_stop * (1 - half_spread)
                    exit_price_kold = kold_stop * (1 - half_spread)
                else:
                    exit_price_boil = boil_stop * (1 + half_spread)
                    exit_price_kold = kold_stop * (1 + half_spread)

                results.append({
                    'entry_date': entry_date,
                    'entry_time': entry_time,
                    'entry_price_boil': entry_price_boil,
                    'exit_price_boil': exit_price_boil,
                    'entry_price_kold': entry_price_kold,
                    'exit_price_kold': exit_price_kold,
                    'reason': 'both_stop'
                })

                in_trade = False
                continue

            # BOIL stops first
            if boil_hit and not kold_hit:
                # Exit BOIL at stop price
                if position == 'long':
                    exit_price_boil = boil_stop - (half_spread*boil_stop)
                else:
                    exit_price_boil = boil_stop + (half_spread*boil_stop)
                
                # Find KOLD exit at EOD - get closest bar at or after exit_time
                kdf = df[df['Date'] >= exit_time].head(1)
                if not kdf.empty:
                    if position == 'long':
                        exit_price_kold = kdf['Close_kold'].iloc[0] - (half_spread*kdf['Close_kold'].iloc[0])
                    else:
                        exit_price_kold = kdf['Close_kold'].iloc[0] + (half_spread*kdf['Close_kold'].iloc[0])
                else:
                    # Fallback to current bar
                    if position == 'long':
                        exit_price_kold = row['Close_kold'] - (half_spread*row['Close_kold'])
                    else:
                        exit_price_kold = row['Close_kold'] + (half_spread*row['Close_kold'])

                results.append({
                    'entry_date': entry_date,
                    'entry_time': entry_time,
                    'entry_price_boil': entry_price_boil,
                    'exit_price_boil': exit_price_boil,
                    'entry_price_kold': entry_price_kold,
                    'exit_price_kold': exit_price_kold,
                    'reason': 'boil_stop'
                })
                in_trade = False
                continue

            # KOLD stops first
            if kold_hit and not boil_hit:
                # Exit KOLD at stop price
                if position == 'long':
                    exit_price_kold = kold_stop - (half_spread*kold_stop)
                else:
                    exit_price_kold = kold_stop + (half_spread*kold_stop)
                
                # Find BOIL exit at EOD - get closest bar at or after exit_time
                bdf = df[df['Date'] >= exit_time].head(1)
                if not bdf.empty:
                    if position == 'long':
                        exit_price_boil = bdf['Close_boil'].iloc[0] - (half_spread*bdf['Close_boil'].iloc[0])
                    else:
                        exit_price_boil = bdf['Close_boil'].iloc[0] + (half_spread*bdf['Close_boil'].iloc[0])
                else:
                    # Fallback to current bar
                    if position == 'long':
                        exit_price_boil = row['Close_boil'] - (half_spread*row['Close_boil'])
                    else:
                        exit_price_boil = row['Close_boil'] + (half_spread*row['Close_boil'])

                results.append({
                    'entry_date': entry_date,
                    'entry_time': entry_time,
                    'entry_price_boil': entry_price_boil,
                    'exit_price_boil': exit_price_boil,
                    'entry_price_kold': entry_price_kold,
                    'exit_price_kold': exit_price_kold,
                    'reason': 'kold_stop'
                })
                in_trade = False
                continue

        # EOD exit
        if in_trade and ts >= exit_time:
            if position == 'long':
                exit_price_boil = row['Close_boil'] - half_spread*row['Close_boil']
                exit_price_kold = row['Close_kold'] - half_spread*row['Close_kold']
            else:
                exit_price_boil = row['Close_boil'] + half_spread*row['Close_boil']
                exit_price_kold = row['Close_kold'] + half_spread*row['Close_kold']

            results.append({
                'entry_date': entry_date,
                'entry_time': entry_time,
                'entry_price_boil': entry_price_boil,
                'exit_price_boil': exit_price_boil,
                'entry_price_kold': entry_price_kold,
                'exit_price_kold': exit_price_kold,
                'reason': 'eod'
            })
            in_trade = False

    trades = pd.DataFrame(results)
    return trades, position


boil = pd.read_csv('data/BOIL_better_data.csv')
kold = pd.read_csv('data/KOLD_better_data.csv')

boil['Date'] = pd.to_datetime(boil['Date'])
kold['Date'] = pd.to_datetime(kold['Date'])

kold['Day'] = kold['Date'].dt.date  # Create date-only column
boil['Day'] = boil['Date'].dt.date  # Create date-only column

rsi_period = 7

boil['RSI'] = RSIIndicator(close=boil['Close'], window=rsi_period).rsi().shift(1)
kold['RSI'] = RSIIndicator(close=kold['Close'], window=rsi_period).rsi().shift(1)

boil['vol_change'] = boil['Volume'].pct_change().shift(1)
kold['vol_change'] = kold['Volume'].pct_change().shift(1)


data = pd.merge(boil, kold, on='Date', suffixes=('_boil', '_kold'))
data['boil_signal'] = 0
data['kold_signal'] = 0

#vol_change and RSI have been shifted by 1, avoiding lookahead bias
data.loc[(data['RSI_boil']<30)&(data['vol_change_boil']>0),'boil_signal'] = 1
data.loc[data['boil_signal']==1,'kold_signal'] = 1


# data.loc[(data['RSI_kold']>70)&(data['vol_change_kold']>0),'kold_signal'] = 1
# data.loc[data['kold_signal']==1,'boil_signal'] = -1


data['Date'] = pd.to_datetime(data['Date'])

data.to_csv('results/hot_and_cold_signal.csv')

trades,pos = make_trade_log(data, stop_loss_pct=0.0005, position='long')#
#0.00025,long,rsi<30
#0.0005,long,rsi<30


trades.to_csv('results.csv')

if trades.empty:
    print("No trades")
else:
    # pos = 'short'  # or read returned position
    if pos == 'long':
        trades['ret_boil'] = (trades.exit_price_boil - trades.entry_price_boil) / trades.entry_price_boil
        trades['ret_kold'] = (trades.exit_price_kold - trades.entry_price_kold) / trades.entry_price_kold
    else:
        trades['ret_boil'] = (trades.entry_price_boil - trades.exit_price_boil) / trades.entry_price_boil
        trades['ret_kold'] = (trades.entry_price_kold - trades.exit_price_kold) / trades.entry_price_kold

    # dollar-equal-weighted pair return = average of the two leg returns
    trades['ret_pair'] = (trades['ret_boil'] + trades['ret_kold']) / 2.0

    # cumulative equity (start at 1)
    trades['cum_equity'] = (1.0 + trades['ret_pair']).cumprod()

    # time-based annualization
    first = pd.to_datetime(trades['entry_date']).min()
    last = pd.to_datetime(trades['entry_date']).max()
    years = max((last - first).days / 365.25, 1/252)  # avoid div by zero
    total_return = trades['ret_pair'].sum()
    ending_equity = trades['cum_equity'].iloc[-1]
    annualized_return = ending_equity ** (1.0 / years) - 1.0

    # annualized Sharpe: scale by sqrt(trades_per_year) if trades are roughly daily,
    # or compute mean/std and scale by sqrt(N_per_year) = sqrt(len(trades) / years)
    trades_per_year = len(trades) / years
    sharpe = (trades['ret_pair'].mean() / trades['ret_pair'].std(ddof=1)) * np.sqrt(trades_per_year)

    print(f"Total return (sum of trade returns): {total_return:.2%}")
    print(f"Ending equity: {ending_equity:.4f}")
    print(f"Ann. return est: {annualized_return:.2%}")
    print(f"Sharpe (annualized est): {sharpe:.2f}")


    print("=== STRATEGY DIAGNOSTICS ===")
    print(f"Number of trades: {len(trades)}")
    print(f"Trades per year: {len(trades) / years:.1f}")
    print(f"Average holding period: {(pd.to_datetime(trades['entry_date']).max() - pd.to_datetime(trades['entry_date']).min()).days / len(trades):.1f} days")

    # Win/Loss Analysis
    trades['is_winner'] = trades['ret_pair'] > 0
    print(f"\nWin rate: {trades['is_winner'].mean():.1%}")
    print(f"Best trade: {trades['ret_pair'].max():.2%}")
    print(f"Worst trade: {trades['ret_pair'].min():.2%}")
    print(f"Avg winning trade: {trades[trades['is_winner']]['ret_pair'].mean():.2%}")
    print(f"Avg losing trade: {trades[~trades['is_winner']]['ret_pair'].mean():.2%}")

    # Check correlation between BOIL and KOLD returns
    boil_returns = data['Close_boil'].pct_change()
    kold_returns = data['Close_kold'].pct_change()
    correlation = boil_returns.corr(kold_returns)
    print(f"\nBOIL-KOLD daily return correlation: {correlation:.3f}")

    # Exit reason breakdown
    print(f"\nExit reasons:")
    print(trades['reason'].value_counts())

    spread = 0.0015/2

    # Transaction cost impact
    total_spread_cost = len(trades) * spread  # spread per round-trip
    print(f"\nTransaction cost analysis:")
    print(f"Total spread cost: {total_spread_cost:.2%}")
    print(f"Returns after spreads: {total_return - total_spread_cost:.2%}")

    # Test if signal matters - compare to random entries
    signal_days = data[data['boil_signal'] == 1].shape[0]
    total_days = data.shape[0]
    print(f"\nSignal frequency: {signal_days}/{total_days} = {signal_days/total_days:.1%} of days")

    # Reality check: Monthly returns
    trades['entry_month'] = pd.to_datetime(trades['entry_date']).dt.to_period('M')
    monthly_returns = trades.groupby('entry_month')['ret_pair'].sum()
    print(f"\nMonthly return consistency:")
    print(f"Positive months: {(monthly_returns > 0).sum()}/{len(monthly_returns)} = {(monthly_returns > 0).mean():.1%}")
    print(f"Best month: {monthly_returns.max():.1%}")
    print(f"Worst month: {monthly_returns.min():.1%}")

    trades['boil_pnl'] = trades['ret_boil'] 
    trades['kold_pnl'] = trades['ret_kold']

    print("Trade-by-trade breakdown (first 10):")
    for i in range(min(10, len(trades))):
        trade = trades.iloc[i]
        print(f"Trade {i+1}: BOIL {trade['boil_pnl']:.2%}, KOLD {trade['kold_pnl']:.2%}, Total {trade['ret_pair']:.2%}")

    print(f"\nAverage individual returns:")
    print(f"BOIL average: {trades['ret_boil'].mean():.3%}")
    print(f"KOLD average: {trades['ret_kold'].mean():.3%}")
    print(f"Both positive: {((trades['ret_boil'] > 0) & (trades['ret_kold'] > 0)).sum()}/{len(trades)}")
    print(f"Both negative: {((trades['ret_boil'] < 0) & (trades['ret_kold'] < 0)).sum()}/{len(trades)}")