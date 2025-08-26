import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


main_df = pd.read_csv('best_pead_results.csv')

main_df['returns'] = ((main_df['exit_price']-main_df['entry_price'])/main_df['entry_price'])*100

plt.scatter(main_df['returns'],main_df[''])
plt.show()
