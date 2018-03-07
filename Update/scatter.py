import pandas as pd
import numpy as np
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt


BTC = pd.read_csv('./bitcoin.csv')
BTC_price =( BTC['close'].values[-30:])[:,np.newaxis]
date = BTC['timeDate'].values[-30:]
DASH = pd.read_csv('./DigitalCash.csv')
DASH_price = (DASH['close'].values[-30:])[:,np.newaxis]
ETH = pd.read_csv('./ethereum.csv')
ETH_price = (ETH['close'].values[-30:])[:,np.newaxis]
LTC = pd.read_csv('./litecoin.csv')
LTC_price = (LTC['close'].values[-30:])[:,np.newaxis]
ZEC = pd.read_csv('./ZCash.csv')
ZEC_price = (ZEC['close'].values[-30:])[:,np.newaxis]

price = np.hstack((BTC_price, DASH_price, ETH_price, LTC_price, ZEC_price))
df = pd.DataFrame(price, index= date, columns= ['BTC', 'DASH', 'ETH', 'LTC', 'ZEC'])
matrix = scatter_matrix(df, alpha= 0.8, figsize= (6, 6), diagonal= 'kde')
for subaxis in matrix:
    for ax in subaxis:
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
plt.suptitle('Scatter matrix of five kinds of CryptoCurrency')
plt.show()
print(df)
