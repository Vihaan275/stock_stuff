import pandas as pd
from scipy.stats import linregress
import numpy as np


def calc_regression_slope(series):
    """
    Calculate regression slope of a pandas Series.
    Returns slope per unit x (per bar).
    """
    y = series.values
    x = range(len(series))
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    return slope

def calculate_KER(price_series, period):
    """
    Calculate Kaufman's Efficiency Ratio (KER) for a pandas Series.
    Returns a pandas Series of KER values.
    """
    # Net change over period (absolute)
    net_change = price_series.diff(period).abs()
    
    # Sum of absolute daily changes over period
    total_change = price_series.diff().abs().rolling(window=period).sum()
    
    # Efficiency Ratio
    ker = net_change / total_change
    
    # Handle division by zero
    ker = ker.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return ker

#EMA Slopes, Kaufman's Efficiency Ratio
#since I have daily bars and will be hopefully doing 1 day to 5 days max of positions, I will be 
#defining a regime as about 20-50 days

price_series = pd.read_csv('spy_adjusted.csv')

regime_filter_num = 20
ema_constant = 10

price_series['ema'] = price_series['Close'].ewm(span=ema_constant,adjust=False).mean().shift(1)
price_series['ema_regression_slope'] = price_series['ema'].rolling(window=regime_filter_num).apply(calc_regression_slope, raw=False)

price_series['KER'] = calculate_KER(price_series['Close'], regime_filter_num).shift(1)

price_series.to_csv('test.csv')


