# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 18:10:29 2018

@author: Topher
"""
#try to predict stock prices with machine learning
#do it with an LSTM - long short term memory NN - apparently good for time series prediction


# load and plot dataset
from pandas import read_csv
from pandas import datetime
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

plt.close('all')


from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler # MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from matplotlib import pyplot
import numpy
import pandas as pd

import stock_funcs

# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df

# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]

# scale train and test data to [-1, 1]
def scale(train, test):
    # fit scaler
    #scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = StandardScaler(with_std=True, with_mean=False)
    scaler = scaler.fit(train)
    # transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    # transform test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled

# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = numpy.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]

# fit an LSTM network to training data
def fit_lstm(train, test, batch_size, nb_epoch, neurons):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    testX, testy = test[:, 0:-1], test[:, -1]
    testX = testX.reshape(testX.shape[0], 1, testX.shape[1])
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
        print('Epoch', i + 1, '/', nb_epoch)
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=1, shuffle=False, validation_data=[testX,testy])
        model.reset_states()
    return model

# make a one-step forecast
def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0,0]


# load dataset
#series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)

num_epochs = 100
num_neurons = 4
batch_size=1






#tickers = ['GLD', 'EEM', 'EFA', 'IYR', 'IYH', 'IYW', 'IYF', 'IYC', 'IYE']
tickers = ['WIKI/AAPL']
#tickers = ['TECL']
tickers.sort()

#tickers = ['TQQQ']

#tickers = ['GLD','VNQ','VEA','VWO','VHT','VFH','VDE','VCR','VGT', 'EEM', 'EFA', 'IYR', 'IYH', 'IYW', 'IYF', 'IYC', 'IYE', 'VTI', 'VEU', 'IEF', 'VNQ', 'GSG']

#tickers = ['VCE','VCN','VDY','VRE','VUN','VUS','VFV','VSP','VGG','VGH','VXC','VIU','VI','VDU','VEF','VE','VEH','VA','VAH','VEE','VAB','VGV','VCB','VSB','VSG','VSC','VLB','VBU','VBG','VVO','VVL','VMO','VLQ','GLD']
num_tickers = len(tickers)
# Define which online source one should use
data_source = 'quandl' #yahoo no longer works

# We would like all available data from 01/01/2000 until 12/31/2016.
start_date_data = '2005-01-01'
end_date_data = '2017-12-31'

adj_close = stock_funcs.get_adjclose( tickers, start_date_data, end_date_data, data_source)

ema_short = adj_close.ewm(span=20, adjust=False).mean()

dates_eom = pd.date_range(start = start_date_data, end=end_date_data, freq='BM') #last day of month

#ema_short = ema_short.loc[dates_eom]


#test it!
plt.close('all')

fig, ax = plt.subplots()
plt.plot(ema_short)

#get the RMSE of the persistence case, where you just predict the same value as today
# split data into train and test
X = ema_short.values
train_size = int(len(ema_short) * 0.8)
train, test = np.split(X, [train_size])

dates_train, dates_test = np.split(dates_eom, [train_size])
# transform data to be stationary
raw_values = ema_short.values
diff_values = np.diff(raw_values, axis=0)
#diff_values = difference(raw_values, 1)

# transform data to be supervised learning
supervised = timeseries_to_supervised(diff_values, 1)
supervised_values = supervised.values

# split data into train and test-sets
train, test = supervised_values[0:train_size], supervised_values[train_size:]

# transform the scale of the data
scaler, train_scaled, test_scaled = scale(train, test)

# fit the model
#lstm_model = fit_lstm(train_scaled, test_scaled, batch_size, num_epochs, num_neurons)
# forecast the entire training dataset to build up state for forecasting
train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
#dum = lstm_model.predict(train_reshaped, batch_size=batch_size)


#test it!

#test it!
plt.close('all')

#fig, ax = plt.subplots()
#plt.plot(dates_eom[0:train_size], raw_values[0:train_size])
#plt.plot(dates_eom[train_size:], raw_values[train_size:])
#
#fig, ax = plt.subplots()
#plt.plot(dates_eom[0:train_size], train_scaled[:,0])
#plt.plot(dates_eom[train_size+1:], test_scaled[:,0])


# slightly smarter trivial case, prediction is just diff between today and yesterday's
predictions = list()
for i in range(len(test_scaled)):
    # make one-step forecast
    X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
    yhat = X[0]
    # invert scaling
    yhat = invert_scale(scaler, X, yhat)
    # invert differencing
    yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
    # store forecast
    predictions.append(yhat)
    #expected = raw_values[len(train) + i + 1]
    #print('Month=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))

actual = raw_values[train_size+1:]

# report performance
rmse = sqrt(mean_squared_error(actual, predictions))
print('Test RMSE to beat: %.3f' % rmse)
# line plot of observed vs predicted
#plt.figure()
#plt.plot(actual)
#plt.plot(predictions)
#plt.show()

#or plot the diffs
predictions = list()
for i in range(len(test_scaled)):
    # make one-step forecast
    X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
    yhat = X[0]
    # invert scaling
    yhat = invert_scale(scaler, X, yhat)
    # invert differencing
    #yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
    # store forecast
    predictions.append(yhat)
    #expected = raw_values[len(train) + i + 1]
    #print('Month=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))

actual = test[:,1]

# report performance
rmse = sqrt(mean_squared_error(actual, predictions))
print('Test RMSE_diff to beat: %.3f' % rmse)
# line plot of observed vs predicted
plt.figure()
plt.plot(actual)
plt.plot(predictions)
plt.show()

# line plot of observed vs predicted
plt.figure()
pyplot.plot(actual, predictions, '.')
pyplot.show()

# walk-forward validation on the train data
predictions = list()
for i in range(len(train_scaled)):
    # make one-step forecast
    X, y = train_scaled[i, 0:-1], train_scaled[i, -1]
    yhat = forecast_lstm(lstm_model, 1, X)
    # invert scaling
    yhat = invert_scale(scaler, X, yhat)
    # invert differencing
    yhat = inverse_difference(raw_values, yhat, -i)
    # store forecast
    predictions.append(yhat)
    #expected = raw_values[len(train) + i + 1]
    #print('Month=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))

# report performance
rmse = sqrt(mean_squared_error(raw_values[0:train_size], predictions))
print('Train RMSE: %.3f' % rmse)
# line plot of observed vs predicted
#plt.figure()
#plt.plot(raw_values[0:train_size])
#plt.plot(predictions)
#plt.show()


# walk-forward validation on the test data
predictions = list()
for i in range(len(test_scaled)):
    # make one-step forecast
    X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
    yhat = forecast_lstm(lstm_model, 1, X)
    # invert scaling
    yhat = invert_scale(scaler, X, yhat)
    # invert differencing
    yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
    # store forecast
    predictions.append(yhat)
    #expected = raw_values[len(train) + i + 1]
    #print('Month=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))

# report performance
rmse = sqrt(mean_squared_error(raw_values[train_size+1:], predictions))
print('Test RMSE: %.3f' % rmse)
# line plot of observed vs predicted
#plt.figure()
#pyplot.plot(raw_values[train_size+1:])
#pyplot.plot(predictions)
#pyplot.show()


predictions = list()
for i in range(len(test_scaled)):
    # make one-step forecast
    X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
    yhat = forecast_lstm(lstm_model, 1, X)
    # invert scaling
    yhat = invert_scale(scaler, X, yhat)
    predictions.append(yhat)

rmse = np.sqrt(mean_squared_error(test[:,-1], predictions))
print('Test RMSE_diffs: %.3f' % rmse)
plt.figure()
pyplot.plot(test[:,-1])
pyplot.plot(np.array(predictions))
pyplot.show()

#plot actual change vs predicted change
plt.figure()
plt.plot(test[:,-1], np.array(predictions), '.' )
plt.plot([-1,1],[-1,1], 'k--')
plt.plot([-1,1],[0,0], 'k--')
plt.plot([0,0],[-1,1], 'k--')
plt.show()
