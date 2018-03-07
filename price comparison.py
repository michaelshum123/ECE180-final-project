import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
import csv
Path='...'
from matplotlib.dates import AutoDateLocator,DateFormatter

#bitcoin data
with open(Path + 'BTC_Last_year_price.csv','r') as f:
    reader = csv.reader(f)
    bitcoin_date = [row[0] for row in reader]
bitcoin_date.pop(0)
with open(Path + 'BTC_Last_year_price.csv','r') as f:
    reader = csv.reader(f)
    bitcoin_price = [row[1] for row in reader]
bitcoin_price.pop(0)
bitcoin_price=[float(i) for i in bitcoin_price]
#ethereum data
with open(Path + 'ETH_Last_year_price.csv','r') as f:
    reader = csv.reader(f)
    ethereum_date = [row[0] for row in reader]
ethereum_date.pop(0)
with open(Path + 'ETH_Last_year_price.csv','r') as f:
    reader = csv.reader(f)
    ethereum_price = [row[1] for row in reader]
ethereum_price.pop(0)
ethereum_price=[float(i) for i in ethereum_price]
#litecoin data
with open(Path + 'LTC_Last_year_price.csv','r') as f:
    reader = csv.reader(f)
    litecoin_date = [row[0] for row in reader]
litecoin_date.pop(0)
with open(Path + 'LTC_Last_year_price.csv','r') as f:
    reader = csv.reader(f)
    litecoin_price = [row[1] for row in reader]
litecoin_price.pop(0)
litecoin_price=[float(i) for i in litecoin_price]
#ZCash data
with open(Path + 'ZEC_Last_year_price.csv','r') as f:
    reader = csv.reader(f)
    zcash_date = [row[0] for row in reader]
zcash_date.pop(0)
with open(Path + 'ZEC_Last_year_price.csv','r') as f:
    reader = csv.reader(f)
    zcash_price = [row[1] for row in reader]
zcash_price.pop(0)
zcash_price=[float(i) for i in zcash_price]
#DASH data
with open(Path + 'DASH_Last_year_price.csv','r') as f:
    reader = csv.reader(f)
    dash_date = [row[0] for row in reader]
dash_date.pop(0)
with open(Path + 'DASH_Last_year_price.csv','r') as f:
    reader = csv.reader(f)
    dash_price = [row[1] for row in reader]
dash_price.pop(0)
dash_price=[float(i) for i in dash_price]

bitcoin_Date, ethereum_Date, litecoin_Date, zcash_Date, dash_Date=[],[],[],[],[]
for i in range(len(bitcoin_date)):
    bitcoin_Date.append(datetime.strptime(bitcoin_date[i],'%Y-%m-%d'))
    ethereum_Date.append(datetime.strptime(ethereum_date[i],'%Y-%m-%d'))
    litecoin_Date.append(datetime.strptime(litecoin_date[i],'%Y-%m-%d'))
    zcash_Date.append(datetime.strptime(zcash_date[i],'%Y-%m-%d'))
    dash_Date.append(datetime.strptime(dash_date[i],'%Y-%m-%d'))


#Plotting

fig, ax = plt.subplots()
ax.plot(bitcoin_Date, bitcoin_price, 'b-',label='BTC')
ax.plot(ethereum_Date, ethereum_price, 'k-',label='ETH')
ax.plot(litecoin_Date, litecoin_price,'r-',label='LTC')
ax.plot(zcash_Date, zcash_price, 'y-',label='ZEC')
ax.plot(dash_Date, dash_price, 'g-',label='DASH')
plt.title('Cryptocurrency Prices (USD)')
plt.ylabel('Coin Value/USD')
plt.yscale('log')
plt.yticks([0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000])
plt.legend()
plt.xticks(rotation=30)
plt.show()
