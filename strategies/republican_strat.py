import yfinance
import pandas as pd
import numpy as np
from datetime import datetime


sp_data = pd.read_csv('data/spy_data_max.csv')
party_years = pd.read_csv('data/presendetial_party_by_year.csv')
sp_data['Date'] = datetime(sp_data['Date'])

sp_data['signal'] = 0

sp_data.loc[sp_data['Date']]