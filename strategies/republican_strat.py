import yfinance
import pandas as pd
import numpy as np
from datetime import datetime


def repub_years(sp_data):
    party_years = pd.read_csv('data/presendetial_party_by_year.csv')
    repub_years = party_years.loc[party_years['Party']=='Republican']
    repub_years_list = repub_years['Year'].to_list()

    sp_data['Date'] = pd.to_datetime(sp_data['Date'])
    sp_data['Year'] = sp_data['Date'].dt.year

    sp_data['signal'] = 0

    sp_data.loc[sp_data['Year'].isin(repub_years_list),'signal'] = 1

    return sp_data