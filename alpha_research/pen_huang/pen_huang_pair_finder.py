import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import yfinance as yf
from itertools import permutations
import csv
from datetime import datetime, timedelta
import pandas_market_calendars as mcal


def polished_dataframe(unpolished,start_date=None,end_date=None):
#CAN REMOVE THIS FUNCTION HERE ONCE WE TURN THE PROJECT INTO A PACKAGE
    unpolished['start_date'] = pd.to_datetime(unpolished['start_date'])
    unpolished['end_date'] = pd.to_datetime(unpolished['end_date'])

    start = start_date
    end = end_date
    
    mask = (unpolished['start_date'] <= end) & (
        unpolished['end_date'].isna() | (unpolished['end_date'] >= start)
    )

    polished_df = unpolished.loc[mask]
    return polished_df


def get_right_stock_data(df,start_date,end_date):


    new_df = df.loc[(df['Date']>=start_date)&(df['Date']<=end_date)]

    return new_df

def ou_log_likelihood(params, x, dt=1):
    mu, theta, sigma = params
    n = len(x) - 1

    xt = x[:-1]
    xt1 = x[1:]

    drift = xt + mu * (theta - xt) * dt
    variance = sigma**2 * dt

    log_likelihood = -0.5 * n * np.log(2 * np.pi * variance) - \
                     0.5 * np.sum(((xt1 - drift) ** 2) / variance)

    return -log_likelihood  # we negate to minimize (MLE maximizes likelihood)

def fit_OU_MLE(x, dt=1):
    # Initial guesses
    mu0 = 1.0
    theta0 = np.mean(x)
    sigma0 = np.std(x)
    
    x = np.array(x)

    result = minimize(ou_log_likelihood, x0=[mu0, theta0, sigma0],
                      args=(x, dt),
                      bounds=[(1e-5, 5), (min(x), max(x)), (1e-5, None)],
                      method='L-BFGS-B')

    if result.success:
        mu, theta, sigma = result.x
        log_likelihood = -result.fun
        return mu, theta, sigma, log_likelihood
    else:
        raise ValueError("MLE optimization failed: " + result.message)
    
def fit_ou_mle_analytical(x, dt=1):
    x = np.array(x)
    x0, x1 = x[:-1], x[1:]
    n = len(x0)

    Sx, Sy = x0.sum(), x1.sum()
    Sxx, Sxy, Syy = (x0*x0).sum(), (x0*x1).sum(), (x1*x1).sum()
    delta = n * Sxx - Sx**2

    # theta (mean-reversion speed)
    theta = -np.log((n * Sxy - Sx * Sy) / delta) / dt

    # mu (long-run mean)
    mu = (Sy - np.exp(-theta * dt) * Sx) / (n * (1 - np.exp(-theta * dt)))

    # sigma (volatility)
    var = (Syy - 2*np.exp(-theta*dt)*Sxy + np.exp(-2*theta*dt)*Sxx) / n
    sigma = np.sqrt(var * 2 * theta / (1 - np.exp(-2*theta*dt)))

    # log-likelihood (fixed sign)
    residuals = x1 - x0 * np.exp(-theta * dt) - mu * (1 - np.exp(-theta * dt))
    ll = (
        -0.5 * n * np.log(2 * np.pi * sigma**2 / (2 * theta))
        - theta / (2 * sigma**2) * np.sum(residuals**2)
    )

    return mu, theta, sigma, ll



def find_best_beta(df1,df2):
    A = 1
    alpha = A/df1.iloc[0]['Close']

    best_log_likelihood = -np.inf
    best_beta = 0
    best_params = (0,0,0)

    for B in np.linspace(0.001,1,10):#start,end,how many total #'s
        beta = B/df2.iloc[0]['Close']
        spread = alpha * df1['Close'].values - beta * df2['Close'].values

        try:#fitting MLE into OU process
            mu, theta, sigma, log_likelihood = fit_ou_mle_analytical(spread)
            if log_likelihood > best_log_likelihood:
                best_log_likelihood = log_likelihood
                best_beta = beta
                best_params = (mu, theta, sigma,alpha)
        except Exception as e:
            print(f"Failed to fit OU for B={B:.3f}: {e}")
    
    return [best_beta,best_log_likelihood,best_params]
        



def potential_pairs(start,end,big_df,sector='Financial Services'):

    unpolished = pd.read_csv('SP500_avoid_survivor_bias.csv')

    #gives list of all tickers that were in the specified time period
    new_df = polished_dataframe(unpolished=unpolished,start_date=start,end_date=end)
    #filters out only finance/banking related stocks
    new_df = new_df[new_df['sector']==sector]




    list1 = new_df['ticker'].tolist()#convert tickers into list

    # Filter big_df to only tickers in list1 once
    filtered_big_df = big_df[big_df['Ticker'].isin(list1)]


    #gives us all combinations of stocks within the list with each item being a tuple
    ticker_combinations = list(permutations(list1,2))

    list2 = []

    for i in ticker_combinations:#(stock1,stock2)
        stock_1 = i[0]
        stock_2 = i[1]

        print(f'Doing for {stock_1} and {stock_2}:',end=' ')

        try:
            df1 = big_df[big_df['Ticker']==stock_1].reset_index(drop=True)
            df2 = big_df[big_df['Ticker']==stock_2].reset_index(drop=True)


            df1 = get_right_stock_data(df1,start_date=start,end_date=end)
            df2 = get_right_stock_data(df2,start_date=start,end_date=end)

            best_beta,best_log_likelihood,(mu,theta,sigma,alpha) = find_best_beta(df1,df2)

            list2 +=[[stock_1,stock_2,best_beta,best_log_likelihood,mu,theta,sigma,alpha]]
            print('Done',end='\n')
        except Exception as e:
            print(f"Couldn't do because of {e}",end='\n')

    #main_df
    df = pd.DataFrame(list2, columns=['Stock1', 'Stock2', 'Best_Beta', 'Best_Log_Likelihood', 'Mu','Theta','Sigma','Alpha'])
    return df


if __name__ == '__main__':
    #this is our out of sample period

    big_df = pd.read_parquet('stock_data/big_boyv2.parquet')

    
    start_date = '2016-01-01'
    end_date = '2018-12-31'
    in_sample_period = 124
    out_of_sample_period = 200
    look_back_period = 60 #look for for how many days back
    num_of_stocks=1

    # Get NYSE calendar
    nyse = mcal.get_calendar('NYSE')
    # Schedule of market opens during your period
    schedule = nyse.schedule(start_date=start_date, end_date=end_date)
    # Convert to a DatetimeIndex of valid trading days
    dates = schedule.index

    
    for out_of_sample_idx in range(in_sample_period,len(dates),out_of_sample_period):

        in_sample_start_idx = out_of_sample_idx-in_sample_period
        #will not include out_of_sample_idx in case of lookahead bias
        #all dates of in_sample_data
        in_sample_dates = dates[in_sample_start_idx:out_of_sample_idx]

        in_sample_start = in_sample_dates[0]
        in_sample_end = in_sample_dates[-1]

        params_df = potential_pairs(start=in_sample_start,end=in_sample_end,big_df=big_df)

        #PAIR FORMATION IS OVER HERE

        out_sample_start_idx = out_of_sample_idx
        out_sample_end_idx = min(out_of_sample_idx + out_of_sample_period, len(dates))
        out_sample_dates = dates[out_sample_start_idx:out_sample_end_idx]

        sorted_params_df = params_df.sort_values(by='Best_Log_Likelihood',ascending=False)

        top_df = sorted_params_df.iloc[0:num_of_stocks]
        print(f'for time period {start_date} to {end_date}, we are executing {top_df.head()}')

        for idx,row in top_df.iterrows():
            stock_1 = row['Stock1']
            stock_2 = row['Stock2']
            beta = row['Best_Beta']
            mu = row['Mu']
            theta = row['Theta']
            sigma = row['Sigma']
            alpha = row['Alpha']

            df1 = get_right_stock_data(big_df, start_date=out_sample_dates[0], end_date=out_sample_dates[-1])
            df2 = get_right_stock_data(big_df, start_date=out_sample_dates[0], end_date=out_sample_dates[-1])

            #this spread is only for out of sample 
            spread = alpha*df1['Close'].values - beta*df2['Close'].values


            position = 0
            pnl = []

            for i in range(look_back_period,len(spread)):
                window = spread[i-look_back_period:i]
    
                # Re-estimate OU parameters daily (Avellaneda and Lee style)
                try:
                    mu_i, theta_i, sigma_i, _ = fit_ou_mle_analytical(window)
                except:
                    continue

                sigma_eq = sigma_i / np.sqrt(2 * mu_i)
                s = (spread[i] - theta_i) / sigma_eq



                



        

    


#basically done with the important work of getting params and doing statistical methods for picking pairs
#just have to start fine tuning it now and make it so that it is physically viable and not just
#statistically viable

























