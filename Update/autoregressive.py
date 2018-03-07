import pandas as pd
from pandas import Series
from pandas import DataFrame
from pandas import concat
from matplotlib import pyplot
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error as mse

data = pd.read_csv('./ECE180-final-project/Update/BTC_last_month_data.csv')
# series = Series.from_csv('../CryptoCurrency/bitcoin.csv', index_col= 3)
series = data['close']
# print(series)
# split dataset
X = series.values
# print(type(X))
# print(X[0])
# train, test = X[1:len(X)-40], X[len(X)-40:]
# # train autoregression
# model = AR(train)
# model_fit = model.fit()
# window = model_fit.k_ar
# coef = model_fit.params
# # walk forward over time steps in test
# history = train[len(train)-window:]
# history = [history[i] for i in range(len(history))]
# predictions = list()
# for t in range(len(test)):
# 	length = len(history)
# 	lag = [history[i] for i in range(length-window,length)]
# 	yhat = coef[0]
# 	for d in range(window):
# 		yhat += coef[d+1] * lag[window-d-1]
# 	obs = test[t]
# 	predictions.append(yhat)
# 	history.append(yhat)
# 	print('predicted=%f, expected=%f' % (yhat, obs))

def AutoRegressive(data, testSize=30, test=True):
	# Autoregressive model used for time-series predictions
	# if test= True, then select the last testSize points as test set
	# else predict for a period of testSize
	# date = np.array(date)
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

pred, error, testData = AutoRegressive(X)
# error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot
pyplot.plot(testData)
pyplot.plot(pred, color='red', label = 'predict')
pyplot.legend()
pyplot.show()
