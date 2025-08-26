import pandas as pd

unpolished = pd.read_csv('SP500_avoid_survivor_bias.csv')


def polished_dataframe(year,unpolished,bruh = True,start_date=None,end_date=None):

    unpolished['start_date'] = pd.to_datetime(unpolished['start_date'])
    unpolished['end_date'] = pd.to_datetime(unpolished['end_date'])

    if bruh:
        start = pd.Timestamp(f'{year}-01-01')
        end = pd.Timestamp(f'{year}-12-31')
    else:
        start = start_date
        end = end_date
    
    mask = (unpolished['start_date'] <= end) & (
        unpolished['end_date'].isna() | (unpolished['end_date'] >= start)
    )

    polished_df = unpolished.loc[mask]
    return polished_df

