import pandas as pd
import matplotlib.pyplot as plt


main_df = pd.read_parquet('results/low_vol_over_reversal.parquet')
print(main_df.head())

main_df['trade_return'] = (main_df['exit_price']-main_df['entry_price'])/main_df['entry_price']

plt.scatter(x=main_df['daily_return'],y=main_df['trade_return'])
plt.show()



#Analysing returns: returns seem to be mostly uniform and centered around 0, with an almost equal amounts
#of bars at the tails too
# plt.hist(main_df['trade_return'], bins=500, edgecolor = 'black')  # Adjust bins as needed
# plt.show()
