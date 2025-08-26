import pandas as pd
import csv
import numpy as np
import ast

df1 = pd.read_csv('results.csv')

#TAKE TOP 1% OF THE PAIRS ACCORDING TO HOW FAST THEY MEAN REVERT

df1['Params'] = df1['Params'].apply(ast.literal_eval)

df1['Mu'] = df1['Params'].apply(lambda x: float(x[0]))
df1['Half_Life'] = np.log(2) / df1['Mu']

# Step 2: Sort by Half_Life
df1_sorted = df1.sort_values(by='Half_Life', ascending=True)

# Step 3: Select top 1% according to smaller half life-> which means faster mean reversion
top_percent = df1_sorted.head(int(0.01 * len(df1_sorted)))


#SIGNAL GENERATION

#grid search for S_o and S_c


