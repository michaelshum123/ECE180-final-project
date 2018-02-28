# import numpy as np
import pandas as pd
# import matplotlib as plt


def data_clean(filename):
    '''
    :type filename: string
    Extract timeDate, close price, open price and volume data
    and write the data to a new csv file
    '''
    df = pd.read_csv('./' + filename + '.csv', sep=',')
    df_clean = df[['timeDate', 'close', 'open', 'volumefrom']]
    df_clean.to_csv(filename + '_clean_data.csv', index=False)


data_clean('bitcoin')
data_clean('litecoin')
data_clean('ethereum')
data_clean('ZCash')
data_clean('DigitalCash')

# BTC data from 2017-01-01 to 2017-12-31
df_BTC = pd.read_csv('bitcoin_clean_data.csv')
BTC_Last_year_data = df_BTC[2361:2726]
BTC_Last_year_data.to_csv('BTC_Last_year_data.csv', index=False)
# BTC data from 2017-06-01 to 2017-12-31
BTC_Half_year_data = df_BTC[2512:2726]
BTC_Half_year_data.to_csv('BTC_Half_year_data.csv', index=False)
# BTC data from 2017-09-01 to 2017-12-31
BTC_Last_quarter_data = df_BTC[2604:2726]
BTC_Last_quarter_data.to_csv('BTC_Last_quarter_data.csv', index=False)
# BTC data from 2017-12-01 to 2017-12-31
BTC_Last_month_data = df_BTC[2695:2726]
BTC_Last_month_data.to_csv('BTC_Last_month_data.csv', index=False)

# LTC data from 2017-01-01 to 2017-12-31
df_LTC = pd.read_csv('litecoin_clean_data.csv')
LTC_Last_year_data = df_LTC[1166:1531]
LTC_Last_year_data.to_csv('LTC_Last_year_data.csv', index=False)
# LTC data from 2017-06-01 to 2017-12-31
LTC_Half_year_data = df_LTC[1317:1531]
LTC_Half_year_data.to_csv('LTC_Half_year_data.csv', index=False)
# LTC data from 2017-09-01 to 2017-12-31
LTC_Last_quarter_data = df_LTC[1409:1531]
LTC_Last_quarter_data.to_csv('LTC_Last_quarter_data.csv', index=False)
# LTC data from 2017-12-01 to 2017-12-31
LTC_Last_month_data = df_LTC[1500:1531]
LTC_Last_month_data.to_csv('LTC_Last_month_data.csv', index=False)

# ETH data from 2017-01-01 to 2017-12-31
df_ETH = pd.read_csv('ethereum_clean_data.csv')
ETH_Last_year_data = df_ETH[514:879]
ETH_Last_year_data.to_csv('ETH_Last_year_data.csv', index=False)
# ETH data from 2017-06-01 to 2017-12-31
ETH_Half_year_data = df_ETH[665:879]
ETH_Half_year_data.to_csv('ETH_Half_year_data.csv', index=False)
# ETH data from 2017-09-01 to 2017-12-31
ETH_Last_quarter_data = df_ETH[757:879]
ETH_Last_quarter_data.to_csv('ETH_Last_quarter_data.csv', index=False)
# ETH data from 2017-12-01 to 2017-12-31
ETH_Last_month_data = df_ETH[848:879]
ETH_Last_month_data.to_csv('ETH_Last_month_data.csv', index=False)

# DASH data from 2017-01-01 to 2017-12-31
df_DASH = pd.read_csv('DigitalCash_clean_data.csv')
DASH_Last_year_data = df_DASH[1059:1424]
DASH_Last_year_data.to_csv('DASH_Last_year_data.csv', index=False)
# DASH data from 2017-06-01 to 2017-12-31
DASH_Half_year_data = df_DASH[1210:1424]
DASH_Half_year_data.to_csv('DASH_Half_year_data.csv', index=False)
# DASH data from 2017-09-01 to 2017-12-31
DASH_Last_quarter_data = df_DASH[1302:1424]
DASH_Last_quarter_data.to_csv('DASH_Last_quarter_data.csv', index=False)
# DASH data from 2017-12-01 to 2017-12-31
DASH_Last_month_data = df_DASH[1393:1424]
DASH_Last_month_data.to_csv('DASH_Last_month_data.csv', index=False)

# ZEC data from 2017-01-01 to 2017-12-31
df_ZEC = pd.read_csv('ZCash_clean_data.csv')
ZEC_Last_year_data = df_ZEC[66:431]
ZEC_Last_year_data.to_csv('ZEC_Last_year_data.csv', index=False)
# ZEC data from 2017-06-01 to 2017-12-31
ZEC_Half_year_data = df_ZEC[217:431]
ZEC_Half_year_data.to_csv('ZEC_Half_year_data.csv', index=False)
# ZEC data from 2017-09-01 to 2017-12-31
ZEC_Last_quarter_data = df_ZEC[309:431]
ZEC_Last_quarter_data.to_csv('ZEC_Last_quarter_data.csv', index=False)
# ZEC data from 2017-12-01 to 2017-12-31
ZEC_Last_month_data = df_ZEC[400:431]
ZEC_Last_month_data.to_csv('ZEC_Last_month_data.csv', index=False)
