import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Strategy Parameters - Easy to modify
SURPRISE_MIN = -80 # Minimum surprise percentage
SURPRISE_MAX = 25   # Maximum surprise percentage
STARTING_CAPITAL = 1000  # Starting capital
MAX_ALLOCATION = 85  # Maximum percentage per stock (if 1 stock gets 100%, if 2 stocks get 50% each)
TRADE_MODE = "buy"  # "buy" or "short" - Change this word to switch between buying and shorting!

# -10 to 25 is best range

class PEADDataStrategy:
    def _init_(self, surprise_min=SURPRISE_MIN, surprise_max=SURPRISE_MAX, max_allocation=MAX_ALLOCATION, trade_mode=TRADE_MODE):
        self.surprise_min = surprise_min
        self.surprise_max = surprise_max
        self.max_allocation = max_allocation
        self.trade_mode = trade_mode.lower()  # "buy" or "short"
        self.starting_capital = STARTING_CAPITAL
        self.trades = []
        
        # Cash management tracking
        self.available_cash = STARTING_CAPITAL
        self.invested_capital = 0
        self.open_positions = []  # Track positions that haven't closed yet
        self.daily_portfolio_values = []
        self.daily_cash_values = []
        self.daily_invested_values = []
        
        # Validate trade mode
        if self.trade_mode not in ["buy", "short"]:
            raise ValueError("trade_mode must be either 'buy' or 'short'")
    
    def load_and_process_data(self):
        """Load the PEAD results data and filter based on our criteria"""
        print("Loading PEAD results data...")
        
        # Load the data
        pead_df = pd.read_csv('pead_results.csv')
        
        print(f"Original PEAD data: {len(pead_df)} trades")
        
        # Clean the data
        pead_df['entry_date'] = pd.to_datetime(pead_df['entry_date'])
        pead_df['exit_date'] = pd.to_datetime(pead_df['exit_date'])
        
        # Filter based on surprise percentage range
        pead_df = pead_df[(pead_df['Surprise(%)'] >= self.surprise_min) & 
                         (pead_df['Surprise(%)'] <= self.surprise_max)]
        
        # Remove any rows with missing data
        pead_df = pead_df.dropna(subset=['entry_price', 'exit_price', 'Surprise(%)'])
        
        print(f"After filtering for surprise range {self.surprise_min}% to {self.surprise_max}%: {len(pead_df)} trades")
        
        # Calculate returns for each trade
        pead_df['profit_per_share'] = pead_df['exit_price'] - pead_df['entry_price']
        pead_df['return_pct'] = (pead_df['profit_per_share'] / pead_df['entry_price']) * 100
        
        # Sort by entry date
        pead_df = pead_df.sort_values('entry_date')
        
        print(f"Date range: {pead_df['entry_date'].min().strftime('%Y-%m-%d')} to {pead_df['exit_date'].max().strftime('%Y-%m-%d')}")
        print(f"Unique symbols: {pead_df['ticker'].nunique()}")
        
        return pead_df
    
    def close_expired_positions(self, current_date):
        """Close positions that have reached their exit date and free up cash"""
        positions_to_close = []
        
        for i, position in enumerate(self.open_positions):
            if position['exit_date'] <= current_date:
                positions_to_close.append(i)
        
        # Close positions in reverse order to avoid index issues
        for i in reversed(positions_to_close):
            position = self.open_positions[i]
            
            # Calculate actual profit based on trade mode
            profit = self.calculate_trade_profit(
                position['shares'], 
                position['entry_price'], 
                position['exit_price']
            )
            
            # Return the exit value to available cash
            exit_value = position['investment'] + profit
            self.available_cash += exit_value
            self.invested_capital -= position['investment']
            
            # Update the trade record with actual profit
            for trade in self.trades:
                if (trade['entry_date'] == position['entry_date'] and 
                    trade['symbol'] == position['symbol'] and
                    abs(trade['investment'] - position['investment']) < 0.01):
                    trade['exit_value'] = exit_value
                    trade['profit'] = profit
                    trade['return_pct'] = (profit / position['investment']) * 100
                    break
            
            # Remove from open positions
            self.open_positions.pop(i)
    
    def calculate_position_size(self, num_trades_today):
        """Calculate position size based on number of trades and available cash"""
        if self.available_cash <= 0 or num_trades_today == 0:
            return 0
        
        # Calculate allocation per trade (max allocation is per stock, so divide by num trades)
        allocation_per_trade = min(self.max_allocation / num_trades_today, 100)
        
        # Use AVAILABLE CASH, not starting capital, for position sizing
        max_investment_per_trade = (allocation_per_trade / 100) * self.available_cash
        
        # This should now be the same as max_investment_per_trade, but keeping for safety
        available_per_trade = self.available_cash / num_trades_today
        
        return min(max_investment_per_trade, available_per_trade)
    
    def calculate_trade_profit(self, shares, entry_price, exit_price):
        """Calculate profit based on trade mode (buy or short)"""
        if self.trade_mode == "buy":
            # Buy low, sell high
            return shares * (exit_price - entry_price)
        else:  # short
            # Sell high, buy back low
            return shares * (entry_price - exit_price)
    
    def run_strategy(self):
        """Run the strategy using existing PEAD data with proper cash management"""
        print("="*60)
        print(f"📊 PEAD STRATEGY WITH PROPER CASH MANAGEMENT - {self.trade_mode.upper()} MODE")
        print("="*60)
        
        # Load the data
        pead_df = self.load_and_process_data()
        
        if len(pead_df) == 0:
            print("❌ No trades found matching criteria!")
            return []
        
        mode_action = "BUYING" if self.trade_mode == "buy" else "SHORTING"
        print(f"\n🚀 Running strategy simulation with {len(pead_df)} historical trades in {mode_action} mode...")
        
        # Get all unique dates to simulate day by day
        all_dates = pd.concat([pead_df['entry_date'], pead_df['exit_date']]).unique()
        all_dates = sorted(all_dates)
        
        # Store dates for plotting
        self.simulation_dates = []
        
        trade_count = 0
        skipped_trades = 0
        
        for current_date in all_dates:
            # First, close any positions that have reached their exit date
            self.close_expired_positions(current_date)
            
            # Then, check for new trades to enter today
            day_trades = pead_df[pead_df['entry_date'] == current_date]
            
            if len(day_trades) > 0:
                num_trades_today = len(day_trades)
                position_size = self.calculate_position_size(num_trades_today)
                
                if position_size > 0:
                    for _, trade in day_trades.iterrows():
                        # Check if we have enough cash for this trade
                        if position_size > self.available_cash:
                            skipped_trades += 1
                            continue
                        
                        # Calculate number of shares we can trade
                        shares = position_size / trade['entry_price']
                        actual_investment = shares * trade['entry_price']
                        
                        # Make sure we don't invest more than we have
                        if actual_investment > self.available_cash:
                            actual_investment = self.available_cash
                            shares = actual_investment / trade['entry_price']
                        
                        if actual_investment < 1:  # Skip tiny trades
                            skipped_trades += 1
                            continue
                        
                        # Deduct from available cash and add to invested capital
                        self.available_cash -= actual_investment
                        self.invested_capital += actual_investment
                        
                        # Create position record
                        position = {
                            'entry_date': trade['entry_date'],
                            'exit_date': trade['exit_date'],
                            'symbol': trade['ticker'],
                            'entry_price': trade['entry_price'],
                            'exit_price': trade['exit_price'],
                            'shares': shares,
                            'investment': actual_investment,
                            'surprise_pct': trade['Surprise(%)']
                        }
                        
                        self.open_positions.append(position)
                        
                        # Create trade record (profit will be calculated when position closes)
                        trade_record = {
                            'entry_date': trade['entry_date'],
                            'exit_date': trade['exit_date'],
                            'symbol': trade['ticker'],
                            'entry_price': trade['entry_price'],
                            'exit_price': trade['exit_price'],
                            'shares': shares,
                            'investment': actual_investment,
                            'exit_value': 0,  # Will be calculated when position closes
                            'profit': 0,      # Will be calculated when position closes
                            'return_pct': 0,  # Will be calculated when position closes
                            'surprise_pct': trade['Surprise(%)'],
                            'trade_mode': self.trade_mode
                        }
                        
                        self.trades.append(trade_record)
                        trade_count += 1
                else:
                    skipped_trades += len(day_trades)
            
            # Record daily portfolio values
            total_portfolio_value = self.available_cash + self.invested_capital
            self.daily_portfolio_values.append(total_portfolio_value)
            self.daily_cash_values.append(self.available_cash)
            self.daily_invested_values.append(self.invested_capital)
            self.simulation_dates.append(current_date)
            
            if trade_count % 100 == 0 and trade_count > 0:
                print(f"Processed {trade_count} trades... Available cash: ${self.available_cash:,.2f}, Invested: ${self.invested_capital:,.2f}")
        
        # Close any remaining open positions at the end
        final_date = max(all_dates)
        self.close_expired_positions(final_date + timedelta(days=30))  # Force close any remaining
        
        print(f"\n✅ Strategy completed!")
        print(f"   Total trades executed: {trade_count} in {self.trade_mode.upper()} mode")
        print(f"   Total trades skipped (insufficient cash): {skipped_trades}")
        print(f"   Final available cash: ${self.available_cash:,.2f}")
        print(f"   Final invested capital: ${self.invested_capital:,.2f}")
        
        return self.trades
    
    def calculate_yearly_stats(self, trades_df):
        """Calculate comprehensive yearly statistics"""
        yearly_stats = []
        
        # Add year column
        trades_df['year'] = trades_df['entry_date'].dt.year
        
        for year in sorted(trades_df['year'].unique()):
            year_trades = trades_df[trades_df['year'] == year].copy()
            
            if len(year_trades) == 0:
                continue
            
            # Basic metrics
            total_trades = len(year_trades)
            winning_trades = len(year_trades[year_trades['profit'] > 0])
            losing_trades = len(year_trades[year_trades['profit'] <= 0])
            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
            
            # Return metrics
            mean_return = year_trades['return_pct'].mean()
            std_return = year_trades['return_pct'].std()
            median_return = year_trades['return_pct'].median()
            
            # Profit metrics
            total_profit = year_trades['profit'].sum()
            avg_win = year_trades[year_trades['profit'] > 0]['profit'].mean() if winning_trades > 0 else 0
            avg_loss = year_trades[year_trades['profit'] <= 0]['profit'].mean() if losing_trades > 0 else 0
            
            # Best and worst trades
            best_trade = year_trades['return_pct'].max() if len(year_trades) > 0 else 0
            worst_trade = year_trades['return_pct'].min() if len(year_trades) > 0 else 0
            
            # Calculate portfolio performance for the year
            yearly_portfolio_return = (total_profit / self.starting_capital) * 100
            
            # Surprise analysis
            positive_surprises = len(year_trades[year_trades['surprise_pct'] > 0])
            negative_surprises = len(year_trades[year_trades['surprise_pct'] <= 0])
            
            yearly_stats.append({
                'Year': year,
                'Total_Trades': total_trades,
                'Winning_Trades': winning_trades,
                'Losing_Trades': losing_trades,
                'Win_Rate_%': win_rate,
                'Mean_Return_%': mean_return,
                'Std_Return_%': std_return,
                'Median_Return_%': median_return,
                'Total_Profit_$': total_profit,
                'Avg_Win_$': avg_win,
                'Avg_Loss_$': avg_loss,
                'Best_Trade_%': best_trade,
                'Worst_Trade_%': worst_trade,
                'Portfolio_Return_%': yearly_portfolio_return,
                'Positive_Surprises': positive_surprises,
                'Negative_Surprises': negative_surprises
            })
        
        return pd.DataFrame(yearly_stats)
    
    def write_results_to_file(self, trades_df):
        """Write comprehensive results to results.txt file"""
        yearly_stats = self.calculate_yearly_stats(trades_df)
        
        filename = f'results_{self.trade_mode}_fixed.txt'
        with open(filename, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"PEAD STRATEGY BACKTEST RESULTS (FIXED CASH MANAGEMENT) - {self.trade_mode.upper()} MODE\n")
            f.write("="*80 + "\n\n")
            
            # Strategy parameters
            f.write("STRATEGY PARAMETERS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Trade Mode: {self.trade_mode.upper()}\n")
            f.write(f"Surprise Range: {self.surprise_min}% to {self.surprise_max}%\n")
            f.write(f"Max Allocation: {self.max_allocation}%\n")
            f.write(f"Starting Capital: ${self.starting_capital:,.2f}\n\n")
            
            # Overall performance
            total_profit = trades_df['profit'].sum()
            total_return = (total_profit / self.starting_capital) * 100
            final_capital = self.available_cash  # All money should be back in cash at the end
            
            winning_trades = trades_df[trades_df['profit'] > 0]
            losing_trades = trades_df[trades_df['profit'] <= 0]
            win_rate = len(winning_trades) / len(trades_df) * 100 if len(trades_df) > 0 else 0
            
            f.write("OVERALL PERFORMANCE:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Trades: {len(trades_df)}\n")
            f.write(f"Total Profit: ${total_profit:,.2f}\n")
            f.write(f"Total Return: {total_return:.2f}%\n")
            f.write(f"Final Capital: ${final_capital:,.2f}\n")
            f.write(f"Overall Win Rate: {win_rate:.2f}%\n")
            f.write(f"Overall Mean Return: {trades_df['return_pct'].mean():.2f}%\n")
            f.write(f"Overall Std Return: {trades_df['return_pct'].std():.2f}%\n\n")
            
            # Cash management info
            f.write("CASH MANAGEMENT:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Final Available Cash: ${self.available_cash:,.2f}\n")
            f.write(f"Final Invested Capital: ${self.invested_capital:,.2f}\n")
            f.write(f"Open Positions Remaining: {len(self.open_positions)}\n\n")
            
            # Yearly breakdown
            f.write("YEARLY BREAKDOWN:\n")
            f.write("="*80 + "\n")
            
            for _, row in yearly_stats.iterrows():
                f.write(f"\n{int(row['Year'])}:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Total Trades: {int(row['Total_Trades'])}\n")
                f.write(f"Winning Trades: {int(row['Winning_Trades'])} ({row['Win_Rate_%']:.2f}%)\n")
                f.write(f"Losing Trades: {int(row['Losing_Trades'])}\n")
                f.write(f"Mean Return: {row['Mean_Return_%']:.2f}%\n")
                f.write(f"Std Return: {row['Std_Return_%']:.2f}%\n")
                f.write(f"Median Return: {row['Median_Return_%']:.2f}%\n")
                f.write(f"Total Profit: ${row['Total_Profit_$']:,.2f}\n")
                f.write(f"Avg Win: ${row['Avg_Win_$']:,.2f}\n")
                f.write(f"Avg Loss: ${row['Avg_Loss_$']:,.2f}\n")
                f.write(f"Best Trade: {row['Best_Trade_%']:.2f}%\n")
                f.write(f"Worst Trade: {row['Worst_Trade_%']:.2f}%\n")
                f.write(f"Portfolio Return: {row['Portfolio_Return_%']:.2f}%\n")
                f.write(f"Positive Surprises: {int(row['Positive_Surprises'])}\n")
                f.write(f"Negative Surprises: {int(row['Negative_Surprises'])}\n")
        
        print(f"✅ Detailed results saved to '{filename}'")
        return yearly_stats
    
    def analyze_results(self):
        """Analyze strategy results"""
        if not self.trades:
            print("No trades to analyze!")
            return None
        
        trades_df = pd.DataFrame(self.trades)
        
        # Calculate key metrics
        total_profit = trades_df['profit'].sum()
        total_return = (total_profit / self.starting_capital) * 100
        final_capital = self.available_cash  # All money should be available cash at the end
        
        winning_trades = trades_df[trades_df['profit'] > 0]
        losing_trades = trades_df[trades_df['profit'] < 0]
        
        win_rate = len(winning_trades) / len(trades_df) * 100 if len(trades_df) > 0 else 0
        avg_win = winning_trades['profit'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['profit'].mean() if len(losing_trades) > 0 else 0
        
        # Print results
        print("\n" + "="*60)
        print(f"PEAD STRATEGY RESULTS (FIXED) - {self.trade_mode.upper()} MODE")
        print("="*60)
        print(f"Strategy Parameters:")
        print(f"  Trade Mode: {self.trade_mode.upper()}")
        print(f"  Surprise Range: {self.surprise_min}% to {self.surprise_max}%")
        print(f"  Max Allocation: {self.max_allocation}%")
        print(f"  Starting Capital: ${self.starting_capital:,.2f}")
        print(f"\nCash Management:")
        print(f"  Final Available Cash: ${self.available_cash:,.2f}")
        print(f"  Final Invested Capital: ${self.invested_capital:,.2f}")
        print(f"  Open Positions: {len(self.open_positions)}")
        print(f"\nPerformance Metrics:")
        print(f"  Total Trades: {len(trades_df)}")
        print(f"  Winning Trades: {len(winning_trades)} ({win_rate:.2f}%)")
        print(f"  Losing Trades: {len(losing_trades)} ({100-win_rate:.2f}%)")
        print(f"  Average Win: ${avg_win:.2f}")
        print(f"  Average Loss: ${avg_loss:.2f}")
        if avg_loss != 0:
            print(f"  Win/Loss Ratio: {abs(avg_win/avg_loss):.2f}")
        print(f"  Total Profit: ${total_profit:,.2f}")
        print(f"  Total Return: {total_return:.2f}%")
        print(f"  Final Capital: ${final_capital:,.2f}")
        
        # Additional statistics
        print(f"\nTrade Statistics:")
        print(f"  Best Trade: ${trades_df['profit'].max():.2f} ({trades_df['return_pct'].max():.2f}%)")
        print(f"  Worst Trade: ${trades_df['profit'].min():.2f} ({trades_df['return_pct'].min():.2f}%)")
        print(f"  Average Trade: ${trades_df['profit'].mean():.2f} ({trades_df['return_pct'].mean():.2f}%)")
        print(f"  Median Trade: ${trades_df['profit'].median():.2f} ({trades_df['return_pct'].median():.2f}%)")
        print(f"  Standard Deviation: {trades_df['return_pct'].std():.2f}%")
        
        # Surprise analysis
        print(f"\nSurprise Analysis:")
        positive_surprise = trades_df[trades_df['surprise_pct'] > 0]
        negative_surprise = trades_df[trades_df['surprise_pct'] < 0]
        
        if len(positive_surprise) > 0:
            print(f"  Positive Surprises: {len(positive_surprise)} trades, Avg Return: {positive_surprise['return_pct'].mean():.2f}%")
        if len(negative_surprise) > 0:
            print(f"  Negative Surprises: {len(negative_surprise)} trades, Avg Return: {negative_surprise['return_pct'].mean():.2f}%")
        
        # Write results to file
        yearly_stats = self.write_results_to_file(trades_df)
        
        return trades_df
    
    def plot_results(self):
        """Create portfolio value visualization"""
        if not self.trades:
            print("No trades to plot!")
            return
        
        # Portfolio Value Over Time with Cash Management Details
        if len(self.daily_portfolio_values) > 0:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
            
            # Choose color based on mode
            line_color = 'blue' if self.trade_mode == 'buy' else 'red'
            mode_label = self.trade_mode.upper()
            
            # Plot 1: Total Portfolio Value
            ax1.plot(self.simulation_dates, self.daily_portfolio_values, linewidth=3, color=line_color, alpha=0.8, 
                    label=f'Total Portfolio Value ({mode_label})')
            ax1.axhline(y=self.starting_capital, color='gray', linestyle='--', linewidth=2, alpha=0.7, 
                       label=f'Starting Capital (${self.starting_capital:,.0f})')
            
            ax1.set_title(f'PEAD Strategy - Portfolio Value Over Time ({mode_label} Mode - Fixed Cash Management)', 
                         fontsize=16, fontweight='bold', pad=20)
            ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
            ax1.grid(True, alpha=0.3)
            ax1.legend(fontsize=12)
            ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            
            # Plot 2: Cash vs Invested Breakdown
            ax2.plot(self.simulation_dates, self.daily_cash_values, linewidth=2, color='green', alpha=0.8, 
                    label='Available Cash')
            ax2.plot(self.simulation_dates, self.daily_invested_values, linewidth=2, color='orange', alpha=0.8, 
                    label='Invested Capital')
            
            ax2.set_title('Cash Management Over Time', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Date', fontsize=12)
            ax2.set_ylabel('Value ($)', fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=12)
            ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            
            # Rotate x-axis labels
            for ax in [ax1, ax2]:
                ax.tick_params(axis='x', rotation=45)
            
            # Add final value annotation
            final_value = self.daily_portfolio_values[-1]
            total_return = ((final_value - self.starting_capital) / self.starting_capital) * 100
            ax1.annotate(f'Final: ${final_value:,.0f}\n({total_return:+.1f}%)', 
                        xy=(self.simulation_dates[-1], final_value),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                        fontsize=10, fontweight='bold')
            
            plt.tight_layout()
            plt.show()
        else:
            print("No portfolio data to plot!")

def main():
    """Main execution function"""
    print("="*60)
    print("🎯 PEAD STRATEGY BACKTEST WITH FIXED CASH MANAGEMENT")
    print("="*60)
    
    # Initialize strategy
    strategy = PEADDataStrategy(
        surprise_min=SURPRISE_MIN,
        surprise_max=SURPRISE_MAX,
        max_allocation=MAX_ALLOCATION,
        trade_mode=TRADE_MODE  # This is where you change "buy" to "short"!
    )
    
    # Run strategy
    trades = strategy.run_strategy()
    
    # Analyze results
    trades_df = strategy.analyze_results()
    
    # Create visualizations
    strategy.plot_results()
    
    # Save results
    if trades_df is not None:
        filename = f'pead_strategy_results_{strategy.trade_mode}_fixed.csv'
        trades_df.to_csv(filename, index=False)
        print(f"\nDetailed results saved to '{filename}'")
        
        # Show some sample trades
        print(f"\nSample of Best Trades ({strategy.trade_mode.upper()} mode):")
        best_trades = trades_df.nlargest(5, 'profit')[['entry_date', 'symbol', 'return_pct', 'profit', 'surprise_pct']]
        print(best_trades.to_string(index=False))
        
        print(f"\nSample of Worst Trades ({strategy.trade_mode.upper()} mode):")
        worst_trades = trades_df.nsmallest(5, 'profit')[['entry_date', 'symbol', 'return_pct', 'profit', 'surprise_pct']]
        print(worst_trades.to_string(index=False))
    
    return strategy, trades_df

if _name_ == '_main_':
    # Run the fixed strategy
    strategy, results = main()