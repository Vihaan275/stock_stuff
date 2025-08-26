import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from statsmodels.tsa.stattools import adfuller, coint

# --- Load and Prepare Data ---
in_sample_start_date = "2016-01-01"
in_sample_end_date = "2017-01-01"

out_of_sample_start = "2017-01-02"
out_of_sample_end = "2018-01-01"

es_data = pd.read_csv('data/es!data.csv')
nq_data = pd.read_csv('data/nq_yahoo.csv')

# Convert to datetime
es_data['date'] = pd.to_datetime(es_data['date'])
nq_data['date'] = pd.to_datetime(nq_data['date'])

# Merge and sort
wide_df = pd.merge(es_data, nq_data, on='date', how='inner', suffixes=('_es', '_nq'))
wide_df = wide_df.sort_values('date')

# --- In-sample Split ---
in_sample = wide_df[(wide_df['date'] >= in_sample_start_date) & (wide_df['date'] <= in_sample_end_date)].copy()
S1 = in_sample['close_es'].values
S2 = in_sample['close_nq'].values

# --- OU Process Log-Likelihood ---
def ou_log_likelihood(params, x):
    mu, theta, sigma = params
    dt = 1
    x_t = x[:-1]
    x_t1 = x[1:]
    # OU transition mean and variance
    m = x_t + mu * (theta - x_t) * dt
    v = sigma**2 * dt
    ll = -0.5 * np.sum(np.log(2 * np.pi * v) + ((x_t1 - m)**2) / v)
    return -ll  # negative log-likelihood for minimization

# --- Grid Search Over Beta ---
B_values = np.linspace(0.001, 1, 100)  # change to 1000 for actual precision
log_likelihoods = []

for B in B_values:
    portfolio = S1 - B * S2
    x = portfolio - np.mean(portfolio)  # de-mean for stability
    
    try:
        res = minimize(ou_log_likelihood, x0=[0.1, 0, 0.1], args=(x,),
                       bounds=[(-5, 5), (np.min(x), np.max(x)), (1e-6, 5)],
                       method='L-BFGS-B')
        log_likelihoods.append(-res.fun)
    except:
        log_likelihoods.append(-np.inf)  # if optimization fails

# --- Find Best Beta ---
best_idx = np.argmax(log_likelihoods)
best_beta = B_values[best_idx]
best_loglik = log_likelihoods[best_idx]

print(f"Best beta: {best_beta:.4f}")
print(f"Log-likelihood: {best_loglik:.2f}")


in_sample['graph'] = in_sample['close_es']-0.3441*in_sample['close_nq']#x_t = ES-Beta*NQ
plt.plot(in_sample['date'],in_sample['graph'])
plt.grid(True)
plt.show()

result = adfuller(in_sample['graph'])
print(f"ADF Statistic: {result[0]}")
print(f"p-value: {result[1]}")


score, pvalue, _ = coint(wide_df['close_es'], wide_df['close_nq'])
print("Cointegration p-value:", pvalue)

#best Beta: 0.3441
 
