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
from scipy.interpolate import interp1d

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
    def __init__(self, date, closeprice, openprice, volume, method='poly', ord=5, testSize=30, test=True):
        super(DataAnalysis, self).__init__()
        self.price = closeprice
        self.oprice = openprice
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
        # self.prediction = self.AutoRegressive(self.price, self.testSize, self.test)
        self.price = [1e5 * p for p in self.price]
    def AutoRegressive(self, data, testSize=2, test=True):
        # Autoregressive model used for time-series predictions
        # if test= True, then select the last testSize points as test set
        # else predict for a period of testSize
        # print(data.shape)
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
        # maxPts, minPts = [], []
        # for i in range(len(self.price)):
        #     if i >= 1 and i <= len(self.price) - 2:
        #         if self.price[i - 1] < self.price[i] > self.price[i + 1]:
        #             maxPts.append(i)
        #         elif self.price[i - 1] > self.price[i] < self.price[i + 1]:
        #             minPts.append(i)
        # sorted(maxPts)
        # sorted(minPts)
        # maxLevel = self.price[maxPts[1]]
        # minLevel = self.price[minPts[-1]]
        # return maxLevel, minLevel

#    def display(self):
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


################################################################################
Path = './' # Please use path based on current dir
Currency = 'BTC'
Period = '_Half_year'
fname = Path + Currency + Period + '_data.csv'
Files = pd.read_csv(fname)
closePrice, openPrice, Volume, date = Files['close'].values, Files['open'].values, Files['volumefrom'].values, Files['timeDate'].values
BTC_Module = DataAnalysis(date, closePrice, openPrice, Volume)

Path = './' # Please use path based on current dir
Currency = 'DASH'
Period = '_Half_year'
fname = Path + Currency + Period + '_data.csv'
Files = pd.read_csv(fname)
closePrice, openPrice, Volume, date = Files['close'].values, Files['open'].values, Files['volumefrom'].values, Files['timeDate'].values
DASH_Module = DataAnalysis(date, closePrice, openPrice, Volume)

Path = './' # Please use path based on current dir
Currency = 'ETH'
Period = '_Half_year'
fname = Path + Currency + Period + '_data.csv'
Files = pd.read_csv(fname)
closePrice, openPrice, Volume, date = Files['close'].values, Files['open'].values, Files['volumefrom'].values, Files['timeDate'].values
ETH_Module = DataAnalysis(date, closePrice, openPrice, Volume)

Path = './' # Please use path based on current dir
Currency = 'LTC'
Period = '_Half_year'
fname = Path + Currency + Period + '_data.csv'
Files = pd.read_csv(fname)
closePrice, openPrice, Volume, date = Files['close'].values, Files['open'].values, Files['volumefrom'].values, Files['timeDate'].values
LTC_Module = DataAnalysis(date, closePrice, openPrice, Volume)

Path = './' # Please use path based on current dir
Currency = 'ZEC'
Period = '_Half_year'
fname = Path + Currency + Period + '_data.csv'
Files = pd.read_csv(fname)
closePrice, openPrice, Volume, date = Files['close'].values, Files['open'].values, Files['volumefrom'].values, Files['timeDate'].values
ZEC_Module = DataAnalysis(date, closePrice, openPrice, Volume)
################################################################################
def monthlyPrice(Module, period):
    assert isinstance(period, int)
    Price = Module.price
    mPrice = []
    for i in range(period):
        mPrice.append(np.mean(Price[30 * i:30 * (i + 1)]))
    return mPrice

def generate():
    monthDays = [30, 31, 31, 30, 31, 30]
    months = [15]
    for i in range(len(monthDays)):
        months.append(months[i] + monthDays[i])
    return months

def separate(data, idx):
    red, redP, green, greenP = [], [], [], []
    for i in idx:
        if data[i] >= 0:
            red.append(data[i])
            redP.append(i)
        else:
            green.append(data[i])
            greenP.append(i)
    return red, redP, green, greenP

priceC = [BTC_Module.price[i + 1] - BTC_Module.price[i] for i in range(len(BTC_Module.price) - 1)]
p_red, p_redP, p_green, p_greenP = separate(priceC, np.arange(len(priceC)))

fig = plt.figure()
ax1 = fig.gca()
n = np.arange(len(BTC_Module.price))
BTC_monthly = monthlyPrice(BTC_Module, 7)
months = generate()
ax1.fill_between(n, BTC_Module.price, len(BTC_Module.price) * [0], facecolor= 'k', alpha= 0.3)
ax1.plot(n, BTC_Module.avePrice, 'r--', label= 'BTC Average Price')
ax1.plot(n, BTC_Module.fitValue, 'b-', label= 'BTC Fitting Curve')
ax1.plot(months, BTC_monthly, 'g-o', label= 'BTC Monthly Averaging')
plt.ylim([0, 20000])
plt.ylabel('Price / USD')
plt.legend()
ax2 = ax1.twinx()
ax2.bar(p_redP, p_red, color= 'r', align= 'edge')
ax2.bar(p_greenP, p_green, color= 'g', align= 'edge')

ax2.grid(True)
plt.title('Price Analysis for BTC')
plt.xlabel('Date')
plt.xticks(BTC_Module.tickPos, BTC_Module.date, rotation= 70)
plt.ylabel('Price / USD')
plt.legend()
plt.show()

################################################################################
def Corr(stat1, stat2, winSize=10):
    corr = []
    corr += winSize * [np.nan]
    stat1 /= np.linalg.norm(stat1)
    stat2 /= np.linalg.norm(stat2)
    for i in range(len(stat1) - winSize):
        data1, data2 = stat1[i:i + winSize], stat2[i:i + winSize]
        iCorr = np.correlate(data1, data2)
        corr.append(iCorr)
    return corr

Path = './' # Please use path based on current dir
Stock = 'DJI'
Period = '_Half_year'
fname = Path + Stock + Period + '_data.csv'
Files = pd.read_csv(fname)
dji_values = Files[[1]].values.flatten()
dji_n = np.arange(len(dji_values))
f_dji = interp1d(dji_n, dji_values)
dji_new_n = np.linspace(0, len(dji_values) - 1, len(n))
dji_int_values = f_dji(dji_new_n)
dji_int_valuesC = [dji_int_values[i+1] - dji_int_values[i] for i in range(len(dji_int_values) - 1)]
d_red, d_redP, d_green, d_greenP = separate(dji_int_valuesC, np.arange(len(dji_int_valuesC)))
dji_corr = Corr(dji_int_valuesC, priceC)
# dji_corr = []
# for i in range(len(n)):
#     iCorr = np.correlate(dji_int_values / np.linalg.norm(dji_int_values), BTC_Module.price / np.linalg.norm(BTC_Module.price))
#     dji_corr.append(iCorr)


Path = './' # Please use path based on current dir
Stock = 'NASDAQ'
Period = '_Half_year'
fname = Path + Stock + Period + '_data.csv'
Files = pd.read_csv(fname)
nas_values = Files[[1]].values.flatten()
nas_n = np.arange(len(nas_values))
f_nas = interp1d(nas_n, nas_values)
nas_new_n = np.linspace(0, len(nas_values) - 1, len(n))
nas_int_values = f_dji(nas_new_n)
nas_int_valuesC = [nas_int_values[i+1] - nas_int_values[i] for i in range(len(nas_int_values) - 1)]
n_red, n_redP, n_green, n_greenP = separate(nas_int_valuesC, np.arange(len(nas_int_valuesC)))

# nas_corr = []
# for i in range(len(n)):
#     iCorr = np.correlate(nas_int_values / np.linalg.norm(dji_int_values), BTC_Module.price / np.linalg.norm(BTC_Module.price))
#     nas_corr.append(iCorr)
nas_corr = Corr(nas_int_valuesC, priceC)
volumeC = [BTC_Module.volume[i+1] - BTC_Module.volume[i] for i in range(len(BTC_Module.volume) - 1)]
v_red, v_redP, v_green, v_greenP = separate(volumeC, np.arange(len(volumeC)))
vol_corr = Corr(volumeC, priceC)
n = np.arange(len(priceC))
fig = plt.subplots()
ax1 = plt.subplot(1,3,1)
ax1.bar(d_redP, d_red, color= 'r', align= 'edge')
ax1.bar(d_greenP, d_green, color= 'g', align= 'edge')
plt.ylabel('Change')
plt.title('DJI VS. Price')
ax2 = ax1.twinx()
ax2.plot(n, dji_corr, 'y-.', linewidth= 1.5,  label= 'Correlation')
plt.xlabel('Date')
plt.xticks(BTC_Module.tickPos, BTC_Module.date, rotation= 70)
plt.legend()
ax3 = plt.subplot(1,3,2)
ax3.bar(n_redP, n_red, color= 'r', align= 'edge')
ax3.bar(n_greenP, n_green, color= 'g', align= 'edge')
plt.ylabel('Changes')
plt.title('NASDAQ VS. Price')
ax4 = ax3.twinx()
ax4.plot(n, nas_corr, 'y-.', linewidth= 1.5,  label= 'Correlation')
plt.xlabel('Date')
plt.xticks(BTC_Module.tickPos, BTC_Module.date, rotation= 70)
plt.legend()
ax5 = plt.subplot(1,3,3)
ax5.bar(v_redP, v_red, color= 'r', align= 'edge')
ax5.bar(v_greenP, v_green, color= 'g', align= 'edge')
plt.ylabel('Values')
plt.title('Volume VS. Price')
ax6 = ax5.twinx()
ax6.plot(n, vol_corr, 'y-.', linewidth= 1.5,  label= 'Correlation')
plt.xlabel('Date')
plt.xticks(BTC_Module.tickPos, BTC_Module.date, rotation= 70)
plt.legend()
plt.show()
plt.suptitle('Correlation with Major Stocks')

# print(BTC_Module.price)
# print(BTC_Module.avePrice)
