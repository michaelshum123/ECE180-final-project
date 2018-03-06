import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import csv
#from time import localtime
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error as mse
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from matplotlib import colors as mcolors


class DataAnalysis(object):
    """
    Including mutiple methods for data analysis:

    AutoRegressive: Price prediction with AR model
        data (float): History price data used for predicting future price
        testSize (int): Time period for prediction
        test (boolean): If True, examine prediction accuracy with test set
                        If False, predict future price for period of testSize

    Corr: Calculate moving correlation between two inputs
        stat1, stat2 (float): data of two variables used for calculating correlation
        winSize (int): Window size for calculating correlation

    dailyChange: Calculate everyday price change percent with respect to openPrice
        openPrice, closePrice (float): price data

    fit: Fitting real data points to a smooth curve
        method (str):
                poly - polynomial curve
                exp  - expotential curve
                sin  - sine curve
                log  - logarithmic curve
        ord (int): polynomial order (other method would ignore this parameter)

    movingStat: Calculating moving average and moving variance of original data
        size (int): Window size used for calculating moving average and variance

    raiseOrfall: Calculate increment or decrement in a period of time
        data (float): Price data
        winSize (int): Time period

    separate: Separate increase data and decrease data and give them different color
        data (float): Price change data
        position (int): start date of each period

    ticks: Generate ticks for xlabel
        tick (str): date
        interval (int): period between two successive ticks

    display: Plot graghs of analysis data
    """
    def __init__(self, date, price, oprice, volume, method='poly', ord=5, testSize=30, test=True):
        super(DataAnalysis, self).__init__()
        self.price = price
        self.oprice = oprice
        self.oridate = date
        self.date, self.tickPos = self.ticks(date, 60)
        self.volume = volume
        self.method = method
        self.ord = ord
        self.testSize = testSize
        self.test = test
        self.avePrice, self.varPrice = self.movingStat(10)
        self.aveVar = self.avePrice + self.varPrice
        self.dailyC = self.dailyChange(self.oprice, self.price)
        self.corr = self.Corr(self.price, self.volume)
        self.fitValue = self.fit(method, self.ord)
        # self.support, self.resistance = self.trendLine()
        self.perc, self.pos = self.raiseOrfall(self.avePrice)
        self.volume = [1e-5 * p for p in self.volume]
        self.r, self.rPos, self.d, self.dPos = self.separate(self.perc, self.pos)
        self.prediction = self.AutoRegressive(self.price, self.testSize, self.test)

    def AutoRegressive(self, data, testSize=2, test=True):
        # Autoregressive model used for time-series predictions
        # if test= True, then select the last testSize points as test set
        # else predict for a period of testSize
        print(data.shape)
        if test:
            trainData = data[:-testSize]
            testData = data[-testSize:]
        else:
            trainData = data

        model = AR(trainData)
        modelFit = model.fit()
        winSize, coeff = modelFit.k_ar, modelFit.params

        predData = list(trainData[-winSize:])
        pred = []
        for i in range(testSize):
            x = list(predData[-winSize:])
            y = coeff[0]
            # use winSize number of data to predict future value
            for n in range(winSize):
                y += coeff[n + 1] * x[winSize - (n + 1)]
            if test:
                # use test data to predict future value
                predData.append(testData[i])
            else:
                # use predicted value to predict future value
                predData.append(y)
            pred.append(y)

        if test:
            error = mse(testData, pred)
            return pred, error, testData
        else:
            error = None
            return pred, error

    def Corr(self, stat1, stat2, winSize=10):
        corr = []
        corr += 10 * [np.nan]
        stat1 /= np.linalg.norm(stat1)
        stat2 /= np.linalg.norm(stat2)
        for i in range(len(stat1) - winSize):
            data1, data2 = stat1[i:i + winSize], stat2[i:i + winSize]
            iCorr = np.correlate(data1, data2)
            corr.append(iCorr)
        return corr

    def dailyChange(self, openPrice, closePrice):
        diff = np.asarray(openPrice) - np.asarray(closePrice)
        perc = diff / np.asarray(openPrice)
        return perc

    def fit(self, method='poly', ord=3):
        # User can customize fit method in 'poly', 'exp', 'sine', 'log'
        # For price, it seems 'poly' works best with high order number
        def funExp(x, a, b):
            return a * np.exp(b * x)

        def funSin(x, a, w, t):
            return a * np.sin(w * x + t)

        def funLog(x, A, a, b):
            return A * np.log(a * x + b)

        data = np.asarray(self.avePrice)[9:]
        n = range(len(data))
        if method == 'poly':
            assert isinstance(ord, int) and ord != 0
            fitCoeff = np.polyfit(n, data, ord)
            fitPoly = np.poly1d(fitCoeff)
            fitValue = fitPoly(n)
        elif method == 'exp':
            fitCoeff, _ = curve_fit(funExp, n, self.avePrice)
            a, b = fitCoeff[0], fitCoeff[1]
            fitValue = funExp(n, a, b)
        elif method == 'sine':
            fitCoeff, _ = curve_fit(funSin, n, self.avePrice)
            a, w, t = fitCoeff[0], fitCoeff[1], fitCoeff[2]
            fitValue = funSin(n, a, w, t)
        elif method == 'log':
            fitCoeff, _ = curve_fit(funLog, n, self.avePrice)
            A, a, b = fitCoeff[0], fitCoeff[1], fitCoeff[2]
            fitValue = funLog(n, A, a, b)
        returnValue = np.zeros((len(fitValue) + 9))
        returnValue[:9] = 9 * np.nan
        returnValue[9:] = fitValue
        return returnValue

    def movingStat(self, size=10):
        data = pd.Series(self.price, np.arange(len(self.price)))
        ave = data.rolling(window=size, center=False).mean()
        var = data.rolling(window=size, center=False).std()
        return ave, var

    def raiseOrfall(self, data, winSize=15):
        # Calculate price change in a period of winSize
        # Change between the start and the end of the period
        # Changing percent with respect to the price at the start
        c, p = [], []
        for i in range(len(data)):
            if i >= len(data) - 15:
                break
            if i % winSize == 0:
                diff = (data[i + 14] - data[i]) / data[i]
                p.append(i)
                c.append(diff)
        return c, p

    def separate(self, data, posit):
        # Separate input data into two sets: positive and negative
        pos, neg = [], []
        posT, negT = [], []
        for i in range(len(data)):
            if data[i] >= 0:
                pos.append(data[i])
                posT.append(posit[i])
            else:
                neg.append(data[i])
                negT.append(posit[i])
        return pos, posT, neg, negT

    def ticks(self, tick, interval=16):
        assert isinstance(interval, int)
        newTick, tickPos = [], []
        for i in range(len(tick)):
            if i % interval == 0:
                newTick.append(tick[i])
                tickPos.append(i)
        return newTick, tickPos

    # def trendLine(self):
        maxPts, minPts = [], []
        for i in range(len(self.price)):
            if i >= 1 and i <= len(self.price) - 2:
                if self.price[i - 1] < self.price[i] > self.price[i + 1]:
                    maxPts.append(i)
                elif self.price[i - 1] > self.price[i] < self.price[i + 1]:
                    minPts.append(i)
        sorted(maxPts)
        sorted(minPts)
        maxLevel = self.price[maxPts[1]]
        minLevel = self.price[minPts[-1]]
        return maxLevel, minLevel

    def display(self):
        plt.figure(1)
        # Price polt
        # Original price, averaged price and its fitting curve
        n = np.arange(len(self.price))
        # supLine = len(n) * [self.support]
        # resLine = len(n) * [self.resistance]
        plt.plot(n, self.price, 'k.-', label='Original price')
        plt.plot(n, self.avePrice, 'r-', label='Moving average')
        plt.plot(n, self.fitValue, 'b-', label='Fitting curve')
        plt.bar(n, self.varPrice, align='center', color='y', label='Moving variance')
        # plt.plot(n, supLine, 'y-', linewidth= 2.0, label= 'Support line')
        # plt.plot(n, resLine, 'g-', linewidth= 2.0, label= 'Resistance line')
        plt.title('Price')
        plt.xticks(self.tickPos, self.date, rotation=70)
        # plt.xticks([])
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.grid(True)
        plt.legend()

        fig, ax1 = plt.subplots()
        # Volume and Price change plot
        # Volume in blue bars
        # Price bars: Red - increase, Green - decrease
        ax1.bar(n, self.volume, align='center', color='b', label='Volume')
        ax1.bar(self.rPos, self.r, 15, align='edge', color='r', alpha=0.7, label='Increase')
        ax1.bar(self.dPos, self.d, 15, align='edge', color='g', alpha=0.7, label='Decrease')
        plt.ylabel('Volume(BTC) / Percent(%)')
        plt.legend()
        plt.grid(True)

        ax2 = ax1.twinx()
        ax2.plot(n, self.corr, 'y-', linewidth=1.7, label='Correlation')
        plt.xticks(self.tickPos, self.date, rotation=70)
        plt.title('Volume and Price Change')
        plt.xlabel('Date')
        plt.legend()
        plt.grid(False)
        fig.tight_layout()
        plt.show()

        plt.subplots()
        plt.hist(self.dailyC, 'auto')
        plt.title('Daily Change Histogram from %s to %s' % (self.date[0], self.date[-1]))
        plt.xlabel('Percent')
        plt.ylabel('Number')
        plt.grid(True)
        plt.show()

        plt.subplots()
        if self.test:
            pred, error, testData = self.prediction
            n = np.arange(len(pred))
            plt.plot(n, testData, 'b-', label='Real price')
            plt.plot(n, pred, 'r-', label='Prediction')
            plt.title('Price prediction with error= %.2f' % error)
            plt.xlabel('Time / days')
            plt.ylabel('Price / USD')
            plt.legend()
            plt.grid(True)
            plt.show()
        else:
            pred, error = self.prediction
            n = np.arange(len(pred))
            plt.plot(n, pred, 'r-', label='Prediction')
            plt.title('Price prediction for %d days' % self.testSize)
            plt.xlabel('Time / days')
            plt.ylabel('Price / USD')
            plt.legend()
            plt.grid(True)
            plt.show()

# Load data:
# data: str, used for x-axis ticks.
# price: float
# volume: float
# oprice: float

Path = './Update/'
Path = './'
fileName = 'bitcoin_clean_data' # example of bitcoin
with open(Path + fileName + '.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    date = [row[0] for row in reader]  # Please input column index of date
with open(Path + fileName + '.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    price = [row[1] for row in reader]  # Please input column index of close
with open(Path + fileName + '.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    volume = [row[3] for row in reader]  # Please input column index of volume
with open(Path + fileName + '.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    oprice = [row[2] for row in reader]  # Please input column index of open

Select the last period percent of the whole dataset.
If using clean datasets, please use full data with period= 1
period = 0.4
start = -1 * int(period * (len(price) - 1))
date = date[start:]
price = [float(p) for p in price[start:]]
volume = [float(v) for v in volume[start:]]
oprice = [float(o) for o in oprice[start:]]

dataAnalysit = DataAnalysis(date, price, oprice, volume, method='poly', ord=10, testSize=30, test=True)
dataAnalysit.display()
############################################3D Visulization Test#####################################################################################
Path = './Update/'
Currency = 'BTC'
Period = 'Half_year'
fname = (Currency + '_' + Period + '_data.csv')

Data = pd.read_csv(Path + fname)
openPrice, closePrice, volume, date = Data['open'].values, Data['close'].values, Data['volumefrom'].values, Data['timeDate'].values
BTC = DataAnalysis(date, closePrice, openPrice, volume)
#################################################################################################################################################
Path = './Update/'
Currency = 'DASH'
Period = 'Half_year'
fname = (Path + Currency + '_' + Period + '_data.csv')

Data = pd.read_csv(fname)
openPrice, closePrice, volume, date = Data['open'].values, Data['close'].values, Data['volumefrom'].values, Data['timeDate'].values
DASH = DataAnalysis(date, closePrice, openPrice, volume)
#################################################################################################################################################
Path = './Update/'
Currency = 'ETH'
Period = 'Half_year'
fname = (Path + Currency + '_' + Period + '_data.csv')

Data = pd.read_csv(fname)
openPrice, closePrice, volume, date = Data['open'].values, Data['close'].values, Data['volumefrom'].values, Data['timeDate'].values
ETH = DataAnalysis(date, closePrice, openPrice, volume)
#################################################################################################################################################
Path = './Update/'
Currency = 'LTC'
Period = 'Half_year'
fname = (Path + Currency + '_' + Period + '_data.csv')

Data = pd.read_csv(fname)
openPrice, closePrice, volume, date = Data['open'].values, Data['close'].values, Data['volumefrom'].values, Data['timeDate'].values
LTC = DataAnalysis(date, closePrice, openPrice, volume)
#################################################################################################################################################
Path = './Update/'
Currency = 'ZEC'
Period = 'Half_year'
fname = (Path + Currency + '_' + Period + '_data.csv')

Data = pd.read_csv(fname)
openPrice, closePrice, volume, date = Data['open'].values, Data['close'].values, Data['volumefrom'].values, Data['timeDate'].values
ZEC = DataAnalysis(date, closePrice, openPrice, volume)
#################################################################################################################################################
def cc(arg):
    return mcolors.to_rgba(arg, alpha=0.6)

BTC_price = BTC.avePrice
DASH_price = DASH.avePrice
ETH_price = ETH.avePrice
LTC_price = LTC.avePrice
ZEC_price = ZEC.avePrice
n = np.arange(len(BTC_price))
z = np.arange(1,6)
print(z)
verts = []
# fig = plt.figure()
# ax = fig.gca()
# path_BTC = ax.fill_between(n, BTC_price, 0).getpaths()[0]
# path_DASH = ax.fill_between(n, DASH_price, 0).getpaths()[0]
# path_ETH = ax.fill_between(n, ETH_price, 0).getpaths()[0]
# path_LTC = ax.fill_between(n, LTC_price, 0).getpaths()[0]
# path_ZCash = ax.fill_between(n, ZCash_price, 0).getpaths()[0]
fig = plt.figure()
ax = fig.gca(projection= '3d')
verts = []
verts.append(list(zip(n, np.log(BTC_price))))
verts.append(list(zip(n, np.log(DASH_price))))
verts.append(list(zip(n, np.log(ETH_price))))
verts.append(list(zip(n, np.log(LTC_price))))
verts.append(list(zip(n, np.log(ZEC_price))))
print(len(verts))
poly = PolyCollection(verts, facecolors=['r', 'g', 'b','y','k'])
poly.set_alpha(0.7)
ax.add_collection3d(poly, zs= z, zdir= 'y')
ax.set_xlabel('Time / day')
ax.set_xlim3d(0, 250)
ax.set_ylabel('CryptoCurrency')
# ax.set_yticks(np.array(z))
# ax.set_yticklabels(np.array(z), ['BTC', 'DASH', 'ETH', 'LTC', 'ZEC'], fontsize= 12)
ax.set_ylim3d(1, 6)
ax.set_zlabel('Price / USD')
ax.set_zlim3d(0, 8)
plt.show()
