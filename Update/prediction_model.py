import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from sklearn.linear_model import BayesianRidge

###############################################################################
'''
Clean Data
Combine needed data to one dataframe
'''
df_btc = pd.read_csv('bitcoin.csv')
df = df_btc[2512:2726][['close', 'volumefrom']]
df = df.set_index(pd.Series(range(df.shape[0])))
DJI = pd.read_csv('DJI_Half_year_data.csv')[' value']
GRLC = pd.read_csv('NASDAQ_Half_year_data.csv')[' value']
LTC = pd.read_csv('LTC_Half_year_price.csv')['close']
DASH = pd.read_csv('DASH_Half_year_price.csv')['close']
ETH = pd.read_csv('ETH_Half_year_price.csv')['close']
ZEC = pd.read_csv('ZEC_Half_year_price.csv')['close']

dji_values = DJI.values
dji_n = np.arange(dji_values.size)
f_dji = interp1d(dji_n, dji_values)
dji_new_n = np.linspace(0, dji_values.size - 1, df.shape[0])
dji_int_values = f_dji(dji_new_n)
DJI = pd.Series(dji_int_values)

grlc_values = GRLC.values
grlc_n = np.arange(grlc_values.size)
f_grlc = interp1d(grlc_n, grlc_values)
grlc_new_n = np.linspace(0, grlc_values.size - 1, df.shape[0])
grlc_int_values = f_grlc(grlc_new_n)
GRLC = pd.Series(grlc_int_values)

df['DJI'] = DJI
df['GRLC'] = GRLC
df['LTC'] = LTC
df['DASH'] = DASH
df['ETH'] = ETH
df['ZEC'] = ZEC

###############################################################################
'''
Train data and plot the comparision between real price and predicted price
'''
label_train = df['close'][7:-23]
feature_train = df.values[:-30]
label_test = df['close'][-7:]
feature_test = df.values[-14:-7]

model = BayesianRidge()
model_fit = model.fit(feature_train, label_train)
pred = model_fit.predict(feature_test)
plt.plot(range(7), label_test, color='black', label='price', linewidth=1.5)
plt.grid(True)
plt.plot(range(7), pred, color='red', label='Prediction', linewidth=1.5)
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('BayesianRidge Model')
plt.legend()
plt.show()
